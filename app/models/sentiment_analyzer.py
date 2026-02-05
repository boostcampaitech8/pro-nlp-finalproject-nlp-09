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

            # 2. 앙상블 모델 예측 실행 (팀원 코드 호출)
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
                return {"error": f"[{commodity}] {target_date} 예측 보고서 생성 실패."}
            
            # 3. LLM Summarizer 호환을 위한 데이터 정규화
            self._enrich_report_for_llm(report, target_date, commodity)
            
            return report
            
        except Exception as e:
            traceback.print_exc()
            return {"error": f"[{commodity}] 시장 예측 중 오류 발생: {str(e)}"}

    def _enrich_report_for_llm(self, report: Dict[str, Any], target_date: str, commodity: str):
        """보고서 데이터를 정규화하여 에이전트가 쓰기 편하게 만듭니다."""
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
    # 간단한 테스트 로직
    pass
