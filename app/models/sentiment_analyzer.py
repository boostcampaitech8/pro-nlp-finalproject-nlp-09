"""
뉴스 감성 분석 및 시장 영향력 예측 모듈
"""

import os
import sys
import json
import traceback
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# BigQuery Client
try:
    from app.utils.bigquery_client import BigQueryClient
except ImportError as e:
    print(f"경고: BigQueryClient 임포트 실패: {e}")
    BigQueryClient = None

# daily_prediction_pipeline (BigQuery에서 가져온 데이터로 예측)
try:
    from app.model.news_sentiment_model.daily_prediction_pipeline import run_daily_prediction as _run_daily_prediction
except ImportError:
    try:
        from model.news_sentiment_model.daily_prediction_pipeline import run_daily_prediction as _run_daily_prediction
    except ImportError:
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
    ) -> Dict[str, Any]:
        """
        특정 품목(commodity)의 뉴스/가격을 BigQuery에서 가져와 일일 예측을 수행합니다.

        Args:
            target_date: 예측 기준 날짜 (YYYY-MM-DD)
            commodity: 상품명 (corn, soybean, wheat)
            lookback_days: 뉴스 조회 일수 (기본 7일)
            model_dir: 모델 디렉토리 경로
            save_file: 결과 JSON 저장 여부

        Returns:
            Dict: 예측 보고서 및 근거 데이터
        """
        if not BigQueryClient:
            return {"error": "BigQueryClient를 사용할 수 없습니다."}
        if not _run_daily_prediction:
            return {"error": "daily_prediction_pipeline을 임포트할 수 없습니다."}

        try:
            # 모델 경로 설정
            if model_dir is None:
                _dir = os.path.dirname(os.path.abspath(__file__))
                model_dir = os.path.join(_dir, "..", "model", "news_sentiment_model", "trained_models_noPrice")
                model_dir = os.path.normpath(model_dir)
            
            bq = BigQueryClient()

            # 1. 품목별 데이터 가져오기 (BigQueryClient가 commodity를 처리함)
            news_df = bq.get_news_for_prediction(
                target_date=target_date,
                lookback_days=lookback_days,
                commodity=commodity,
                filter_status="T" # 기본적으로 필터링된 뉴스만 사용
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
                return {"error": f"[{commodity}] {target_date} 예측 데이터 생성 실패."}
            
            # 3. LLM Summarizer 호환을 위한 데이터 가공
            self._enrich_report_for_llm(report, target_date, commodity)
            
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