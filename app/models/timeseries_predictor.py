"""
시계열 예측 모듈

BigQuery에서 데이터를 가져와 시계열 모델로 시장 추세를 예측합니다.
"""

from libs.gcp import get_bq_service
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

# from libs.gcp import GCPServiceFactory
# from libs.utils.config import get_config

logger = logging.getLogger(__name__)

# 모델 학습 시 사용된 피처 이름과 BigQuery 컬럼명 매핑
# BigQuery는 소문자를 반환하지만, 모델은 대문자로 학습됨
FEATURE_COLUMN_MAPPING = {
    "volume": "Volume",
    "ema": "EMA",
    "volume_lag1": "Volume_lag1",
    "ema_lag1": "EMA_lag1",
    "volume_lag1_effect": "Volume_lag1_effect",
    "ema_lag1_effect": "EMA_lag1_effect",
}


def normalize_feature_columns(df):
    """
    BigQuery에서 반환된 DataFrame의 컬럼명을 모델이 기대하는 형태로 변환합니다.

    모델 학습 시 사용된 피처 이름(대문자)과 BigQuery 컬럼명(소문자) 간의
    불일치를 해결합니다.

    Args:
        df: BigQuery에서 조회한 DataFrame

    Returns:
        컬럼명이 변환된 DataFrame
    """
    rename_dict = {k: v for k, v in FEATURE_COLUMN_MAPPING.items() if k in df.columns}

    if rename_dict:
        logger.debug(f"[normalize_feature_columns] 컬럼명 변환: {rename_dict}")
        df = df.rename(columns=rename_dict)
    else:
        logger.debug("[normalize_feature_columns] 변환할 컬럼 없음")

    return df


# TimeSeriesInference 모듈 임포트 (지연 로딩)
try:
    from app.model.timeseries_model.inference import TimeSeriesInference
except ImportError as e:
    logger.warning(f"TimeSeriesInference를 임포트할 수 없습니다: {e}")
    TimeSeriesInference = None

# from app.model.timeseries_model.xg_inference import TimeSeriesXGBoostInference

# 모듈 수준 캐싱
_inference_engine = None
_bq_service = None




def get_inference_engine():
    """TimeSeriesInference 인스턴스 (싱글톤)"""
    global _inference_engine
    logger.debug("[get_inference_engine] 호출됨")
    if _inference_engine is None:
        logger.debug("[get_inference_engine] 새 인스턴스 생성 시작")
        if TimeSeriesInference is None:
            logger.error("[get_inference_engine] TimeSeriesInference 모듈이 None임")
            raise ImportError("TimeSeriesInference 모듈을 사용할 수 없습니다.")
        _inference_engine = TimeSeriesInference()
        logger.info("[get_inference_engine] TimeSeriesInference 인스턴스 생성 완료")
    else:
        logger.debug("[get_inference_engine] 기존 캐시된 인스턴스 반환")
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
    logger.info(
        f"[predict_market_trend] 시작 - target_date: {target_date}, commodity: {commodity}"
    )
    # TODO try except 형태에서 return 하지 말고 throw 형태로 변경
    # 날짜 형식 검증
    logger.debug(f"[predict_market_trend] 날짜 형식 검증 중: {target_date}")
    try:
        _ = datetime.strptime(target_date, "%Y-%m-%d")
        logger.debug(f"[predict_market_trend] 날짜 형식 검증 성공")
    except ValueError as e:
        logger.error(
            f"[predict_market_trend] 날짜 형식 검증 실패: {target_date}, error: {e}"
        )
        return json.dumps(
            {
                "error": f"잘못된 날짜 형식입니다: '{target_date}'. YYYY-MM-DD 형식을 사용해주세요."
            },
            ensure_ascii=False,
        )

    # 추론 엔진 가져오기
    logger.debug(f"[predict_market_trend] 추론 엔진 초기화 시작")
    try:
        engine = get_inference_engine()
        logger.info(
            f"[predict_market_trend] 추론 엔진 초기화 성공: {type(engine).__name__}"
        )
    except ImportError as e:
        logger.error(f"[predict_market_trend] 추론 엔진 초기화 실패: {e}")
        return json.dumps(
            {"error": f"추론 엔진 초기화 실패: {str(e)}"}, ensure_ascii=False
        )

    # BigQuery에서 데이터 가져오기
    # TODO 90일치로 고정된 부분 반드시 config로 수정
    # TODO bigquery 조회 전혀 못하고 있는 중
    logger.debug(f"[predict_market_trend] BigQuery 데이터 조회 시작")
    try:
        bq = get_bq_service()
        logger.debug(
            f"[predict_market_trend] BigQuery 서비스 획득 성공: {type(bq).__name__}"
        )

        # Prophet 피처 조회 (타겟 날짜 기준 90일치)
        # 7일 평균 통계와 추세 문맥을 충분히 확보하기 위해 90일치를 조회합니다.
        logger.info(
            f"[predict_market_trend] Prophet 피처 조회 중 - commodity: {commodity}, target_date: {target_date}, lookback_days: 90"
        )
        history_df = bq.get_prophet_forecast_features(
            commodity=commodity,
            target_date=target_date,
            lookback_days=90,
        )
        logger.debug(
            f"[predict_market_trend] Prophet 피처 조회 완료 - 행 개수: {len(history_df)}, 열: {list(history_df.columns) if not history_df.empty else 'N/A'}"
        )

        # history_df = bq.get_daily_prices(
        #     commodity="corn",
        #     target_date=target_date,
        #     lookback_days=90,
        # )

        if history_df.empty:
            logger.warning(
                f"[predict_market_trend] BigQuery 조회 결과가 비어있음 - target_date: {target_date}, commodity: {commodity}"
            )
            return json.dumps(
                {
                    "error": f"BigQuery에서 {target_date} (및 이전)에 대한 데이터를 찾을 수 없습니다."
                },
                ensure_ascii=False,
            )

        logger.info(
            f"[predict_market_trend] BigQuery 데이터 조회 성공 - {len(history_df)} rows for {target_date}"
        )
        logger.debug(
            f"[predict_market_trend] 데이터프레임 샘플 (처음 3행):\n{history_df.head(3).to_string()}"
        )

        # 컬럼명을 모델이 기대하는 형태로 변환
        history_df = normalize_feature_columns(history_df)
        logger.debug(
            f"[predict_market_trend] 컬럼명 변환 후: {list(history_df.columns)}"
        )

    except Exception as e:
        logger.exception("BigQuery 데이터 조회 실패")
        return json.dumps(
            {"error": f"BigQuery 데이터 조회 실패: {str(e)}"}, ensure_ascii=False
        )

    # 예측 수행
    logger.info(
        f"[predict_market_trend] 예측 시작 - target_date: {target_date}, 입력 데이터 크기: {history_df.shape}"
    )
    try:
        result = engine.predict(history_df, target_date)
        logger.info(
            f"[predict_market_trend] 예측 성공 - 결과 키: {list(result.keys()) if isinstance(result, dict) else type(result)}"
        )
        logger.debug(
            f"[predict_market_trend] 예측 결과 상세:\n{json.dumps(result, ensure_ascii=False, indent=2)}"
        )

        # LLM용 출력 포맷팅
        final_result = json.dumps(result, ensure_ascii=False)
        logger.info(
            f"[predict_market_trend] 완료 - 결과 길이: {len(final_result)} chars"
        )
        return final_result

    except ValueError as ve:
        logger.error(f"[predict_market_trend] ValueError 발생: {ve}")
        return json.dumps({"error": str(ve)}, ensure_ascii=False)
    except Exception as e:
        logger.exception(f"[predict_market_trend] 예측 중 예기치 않은 오류 발생: {e}")
        return json.dumps(
            {"error": f"예측 중 예기치 않은 오류가 발생했습니다: {str(e)}"},
            ensure_ascii=False,
        )


# if __name__ == "__main__":
#     # 함수 테스트 (BQ 자격 증명 필요)
#     logging.basicConfig(level=logging.INFO)
#     print("'2025-11-26' 날짜로 predict_market_trend 테스트 중...")
#     print(predict_market_trend("2025-11-26"))
