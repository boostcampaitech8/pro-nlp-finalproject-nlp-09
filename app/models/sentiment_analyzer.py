"""
뉴스 감성 분석 및 시장 영향력 예측 모듈
"""

import os
import sys
import json
import traceback
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

# 프로젝트 루트 및 모델 디렉토리를 Python 경로에 추가 (임포트 에러 해결)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
model_dir_path = os.path.join(project_root, "app", "model", "news_sentiment_model")

if project_root not in sys.path:
    sys.path.append(project_root)
if model_dir_path not in sys.path:
    sys.path.append(model_dir_path)

# BigQuery Client
try:
    from app.utils.bigquery_client import BigQueryClient
except ImportError as e:
    print(f"경고: BigQueryClient 임포트 실패: {e}")
    BigQueryClient = None

# daily_prediction_pipeline 임포트
try:
    # sys.path 추가 덕분에 내부의 ensemble_predictor 등을 찾을 수 있게 됨
    from app.model.news_sentiment_model.daily_prediction_pipeline import run_daily_prediction as _run_daily_prediction
except ImportError as e:
    print(f"경고: daily_prediction_pipeline 임포트 실패 ({e}). 경로 확인 필요: {model_dir_path}")
    _run_daily_prediction = None


class SentimentAnalyzer:
    """
    뉴스 감성 분석 및 시장 예측기.
    """

    def __init__(self):
        pass

    def _dedupe_news_by_text_prefix(self, news_df, text_col: str = "all_text", prefix_length: int = 50):
        """
        all_text(또는 지정 컬럼) 기준 앞 prefix_length자(공백 정규화 후)가 같은 행은 중복으로 보고,
        publish_date 기준 가장 최근 행만 남깁니다.
        """
        try:
            import pandas as pd
        except ImportError:
            return news_df
        if news_df.empty:
            return news_df
        # 텍스트 컬럼: all_text 없으면 description 사용
        if text_col not in news_df.columns:
            text_col = "description" if "description" in news_df.columns else news_df.columns[0]
        col = news_df[text_col]
        # 공백 정규화 후 앞 prefix_length자
        prefix = col.fillna("").astype(str).str.split().str.join(" ").str.slice(0, prefix_length)
        news_df = news_df.copy()
        news_df["_text_prefix"] = prefix
        # 날짜 컬럼: 문자열이면 파싱해서 정렬
        date_col = "publish_date"
        if date_col not in news_df.columns:
            return news_df.drop(columns=["_text_prefix"], errors="ignore")
        df_sorted = news_df.sort_values(date_col, ascending=False)
        df_deduped = df_sorted.drop_duplicates(subset=["_text_prefix"], keep="first")
        return df_deduped.drop(columns=["_text_prefix"], errors="ignore").reset_index(drop=True)

    def run_daily_prediction(
        self,
        target_date: str,
        commodity: str = "corn",
        lookback_days: int = 7,
        model_dir: Optional[str] = None,
        save_file: bool = False,
        **kwargs # 예상치 못한 인자(filter_status 등) 처리용
    ) -> Dict[str, Any]:
        """
        특정 품목(commodity)의 뉴스/가격을 BigQuery에서 가져와 일일 예측을 수행합니다.
        """
        if not BigQueryClient:
            return {"error": "BigQueryClient를 사용할 수 없습니다."}
        if not _run_daily_prediction:
            return {"error": "daily_prediction_pipeline을 임포트할 수 없습니다. 경로 설정을 확인하세요."}

        try:
            # 앙상블 모델 경로 설정
            if model_dir is None:
                # app/model/news_sentiment_model/trained_models 디렉토리 사용
                model_dir = os.path.join(model_dir_path, "trained_models")
                if not os.path.exists(model_dir):
                    # 차선책: trained_models_noPrice 확인
                    alt_model_dir = os.path.join(model_dir_path, "trained_models_noPrice")
                    if os.path.exists(alt_model_dir):
                        model_dir = alt_model_dir

            bq = BigQueryClient()

            # 1. 품목별 데이터 가져오기
            news_df = bq.get_news_for_prediction(
                target_date=target_date,
                lookback_days=lookback_days,
                commodity=commodity
            )
            price_df = bq.get_price_history(
                target_date=target_date, 
                lookback_days=30, 
                commodity=commodity
            )

            if news_df.empty:
                return {"error": f"[{commodity}] {target_date} 기준 최근 뉴스 데이터가 없습니다."}
            if price_df.empty:
                return {"error": f"[{commodity}] {target_date} 기준 최근 가격 데이터가 없습니다."}

            # 1-1. all_text 기준 앞 50자 동일 시 중복 제거 (가장 최근 것만 유지)
            news_df = self._dedupe_news_by_text_prefix(news_df, prefix_length=50)

            # 2. 앙상블 모델 예측 실행
            report = _run_daily_prediction(
                target_date=target_date,
                lookback_days=lookback_days,
                news_df=news_df,
                price_df=price_df,
                model_dir=model_dir,
                output_dir="outputs",
                save_file=save_file,
            )
            
            if report is None:
                return {"error": f"[{commodity}] {target_date} 예측 데이터 준비 실패 (뉴스/임베딩 부족)."}
            
            # 3. LLM Summarizer 호환을 위한 데이터 가공
            self._enrich_report_for_llm(report, target_date, commodity)
            
            # llm_summarizer 호환: evidence_news = supporting (최대 3개) + opposing (최대 3개)
            evidence = report.get("evidence") or {}
            supporting = evidence.get("supporting_news") or []
            opposing = evidence.get("opposing_news") or []
            evidence_news = [
                {
                    "title": x.get("title", ""),
                    # all_text 우선 사용 (없으면 description 폴백)
                    "all_text": x.get("all_text") or x.get("description", ""),
                    "impact_score": x.get("impact_score"),
                    "sentiment": x.get("sentiment", ""),
                }
                for x in (supporting[:3] + opposing[:3])
            ]
            report["evidence_news"] = evidence_news
            return report
            
        except Exception as e:
            traceback.print_exc()
            return {"error": f"[{commodity}] 시장 예측 중 오류 발생: {str(e)}"}

    def _enrich_report_for_llm(self, report: Dict[str, Any], target_date: str, commodity: str):
        """보고서에 LLM 에이전트가 필요로 하는 필드를 추가 및 정규화합니다."""
        evidence = report.get("evidence") or {}
        supporting = evidence.get("supporting_news") or []
        opposing = evidence.get("opposing_news") or []
        
        # 근거 뉴스 리스트 통합 및 정렬 (상위 5개)
        combined = [
            {
                "title": x.get("title", ""),
                "all_text": x.get("description", ""),
                "price_impact_score": x.get("impact_score", 0.0),
                "sentiment": x.get("sentiment", ""),
            }
            for x in (supporting + opposing)
        ]
        combined.sort(key=lambda a: abs(a.get("price_impact_score") or 0), reverse=True)
        
        report["evidence_news"] = combined[:5]
        report["target_date"] = target_date
        report["commodity"] = commodity


if __name__ == "__main__":
    # 실행 테스트 로직 (필요 시)
    pass
