"""
시계열 예측 모듈

BigQuery에서 데이터를 가져와 시계열 모델로 시장 추세를 예측합니다.
"""

import sys
import os
import json
import logging
from datetime import datetime

# 프로젝트 루트 경로 설정
# current_dir = os.path.dirname(os.path.abspath(__file__))
# root_dir = os.path.dirname(os.path.dirname(current_dir))
# if root_dir not in sys.path:
#     sys.path.append(root_dir)

from libs.gcp import GCPServiceFactory
from libs.utils.config import get_config

logger = logging.getLogger(__name__)


# TimeSeriesInference 모듈 임포트 (지연 로딩)
try:
    from model.timeseries_model.inference import TimeSeriesInference
except ImportError as e:
    logger.warning(f"TimeSeriesInference를 임포트할 수 없습니다: {e}")
    TimeSeriesInference = None


# 모듈 수준 캐싱
_inference_engine = None
_bq_service = None


def _get_bq_service():
    """BigQueryService 인스턴스 (싱글톤)"""
    global _bq_service
    if _bq_service is None:
        config = get_config()
        factory = GCPServiceFactory()
        _bq_service = factory.get_bigquery_client(dataset_id=config.bigquery.dataset_id)
        logger.debug("BigQueryService initialized")
    return _bq_service


def get_inference_engine():
    """TimeSeriesInference 인스턴스 (싱글톤)"""
    global _inference_engine
    if _inference_engine is None:
        if TimeSeriesInference is None:
            raise ImportError("TimeSeriesInference 모듈을 사용할 수 없습니다.")
        _inference_engine = TimeSeriesInference()
        logger.debug("TimeSeriesInference initialized")
    return _inference_engine


def predict_market_trend(target_date: str, commodity: str) -> str:
    """
    시계열 모델을 사용하여 특정 날짜의 금융 시장 추세를 예측합니다.
    BigQuery에서 필요한 피처 데이터를 가져옵니다.

    Args:
        target_date: 분석할 날짜 ('YYYY-MM-DD' 형식)

    Returns:
        str: 상세 예측 지표를 포함한 JSON 형식의 문자열

    Example:
        >>> result = predict_market_trend("2025-01-31")
        >>> data = json.loads(result)
    """
    # TODO try except 형태에서 return 하지 말고 throw 형태로 변경
    # 날짜 형식 검증
    try:
        _ = datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        return json.dumps(
            {
                "error": f"잘못된 날짜 형식입니다: '{target_date}'. YYYY-MM-DD 형식을 사용해주세요."
            },
            ensure_ascii=False,
        )

    # 추론 엔진 가져오기
    try:
        engine = get_inference_engine()
    except ImportError as e:
        return json.dumps(
            {"error": f"추론 엔진 초기화 실패: {str(e)}"}, ensure_ascii=False
        )

    # BigQuery에서 데이터 가져오기
    # TODO 90일치로 고정된 부분 반드시 config로 수정
    # TODO bigquery 조회 전혀 못하고 있는 중
    try:
        bq = _get_bq_service()

        # Prophet 피처 조회 (타겟 날짜 기준 90일치)
        # 7일 평균 통계와 추세 문맥을 충분히 확보하기 위해 90일치를 조회합니다.
        # history_df = bq.get_prophet_forecast_features(
        #     commodity=commodity,
        #     target_date=target_date,
        #     lookback_days=90,
        # )

        history_df = bq.get_daily_prices(
            commodity="corn",
            target_date=target_date,
            lookback_days=90,
        )

        if history_df.empty:
            return json.dumps(
                {
                    "error": f"BigQuery에서 {target_date} (및 이전)에 대한 데이터를 찾을 수 없습니다."
                },
                ensure_ascii=False,
            )

        logger.info(f"Retrieved {len(history_df)} rows for {target_date}")

    except Exception as e:
        logger.exception("BigQuery 데이터 조회 실패")
        return json.dumps(
            {"error": f"BigQuery 데이터 조회 실패: {str(e)}"}, ensure_ascii=False
        )

    # 예측 수행
    try:
        result = engine.predict(history_df, target_date)

        # LLM용 출력 포맷팅
        return json.dumps(result, ensure_ascii=False)

    except ValueError as ve:
        return json.dumps({"error": str(ve)}, ensure_ascii=False)
    except Exception as e:
        logger.exception("예측 중 오류 발생")
        return json.dumps(
            {"error": f"예측 중 예기치 않은 오류가 발생했습니다: {str(e)}"},
            ensure_ascii=False,
        )


# if __name__ == "__main__":
#     # 함수 테스트 (BQ 자격 증명 필요)
#     logging.basicConfig(level=logging.INFO)
#     print("'2025-11-26' 날짜로 predict_market_trend 테스트 중...")
#     print(predict_market_trend("2025-11-26"))
