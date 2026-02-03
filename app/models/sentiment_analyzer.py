"""
뉴스 감성 분석 모듈

BigQuery에서 뉴스 및 가격 데이터를 가져와 시장 영향력을 예측합니다.
"""

import logging
from typing import Dict, Any

from libs.gcp.provider import get_bq_service
import traceback

logger = logging.getLogger(__name__)

# 1. 뉴스 모델 컴포넌트 임포트 (지연 로딩)
try:
    from app.model.news_sentiment_model.inference_with_evidence import (
        CornPricePredictor,
    )
    from app.model.news_sentiment_model.preprocessing import preprocess_news_data
except ImportError as e:
    logger.warning(f"뉴스 감성 모델 모듈 임포트 실패: {e}")
    CornPricePredictor = None
    preprocess_news_data = None

# 2. BigQuery 클라이언트 임포트
try:
    from app.utils.bigquery_client import BigQueryClient
except ImportError as e:
    print(f"경고: BigQueryClient 임포트 실패: {e}")
    BigQueryClient = None

from libs.utils import config


class SentimentAnalyzer:
    """
    뉴스 감성 분석 및 시장 영향력 예측기

    BigQuery에서 뉴스와 가격 데이터를 가져와 시장 영향을 분석합니다.

    Attributes:
        predictor: CornPricePredictor 인스턴스
        _bq: BigQueryService 인스턴스

    Example:
        >>> analyzer = SentimentAnalyzer()
        >>> result = analyzer.predict_market_impact("2025-01-31")
    """

    def __init__(self, commodity: str = "corn"):
        """SentimentAnalyzer 초기화"""
        self.commodity = commodity
        self.predictor = None
        self._initialize_models()
        logger.info(f"[SentimentAnalyzer.__init__] 초기화 완료")

    def _initialize_models(self):
        """모델 및 서비스 초기화"""
        logger.debug("[SentimentAnalyzer._initialize_models] 모델 초기화 시작")

        # TODO 품종별 확장 반드시 고려할 것
        try:
            if CornPricePredictor:
                logger.debug(
                    "[SentimentAnalyzer._initialize_models] CornPricePredictor 인스턴스 생성 중"
                )
                self.predictor = CornPricePredictor()
                self.predictor.load_model()
                logger.info(
                    "[SentimentAnalyzer._initialize_models] CornPricePredictor 초기화 및 모델 로드 완료"
                )
            else:
                logger.warning(
                    "[SentimentAnalyzer._initialize_models] CornPricePredictor 모듈이 None - 임포트 실패 상태"
                )
        except Exception as e:
            logger.error(
                f"[SentimentAnalyzer._initialize_models] 모델 초기화 중 오류 발생: {e}",
                exc_info=True,
            )

    def predict_market_impact(self, target_date: str) -> Dict[str, Any]:
        """
        특정 날짜의 뉴스 기반 시장 예측 (근거 뉴스 포함)

        Args:
            target_date: 분석할 날짜 (YYYY-MM-DD)

        Returns:
            Dict[str, Any]: 예측 결과 및 근거 뉴스 리스트
                - prediction: 예측 값
                - probability: 상승 확률
                - evidence_news: 근거 뉴스 리스트

        Example:
            >>> result = analyzer.predict_market_impact("2025-01-31")
            >>> print(result["probability"])
        """
        logger.info(
            f"[predict_market_impact] 시작 - target_date: {target_date}, commodity: {self.commodity}"
        )

        if not self.predictor:
            logger.error("[predict_market_impact] 뉴스 예측 모델이 로드되지 않음")
            return {"error": "뉴스 예측 모델이 로드되지 않았습니다."}

        # if not BigQueryClient:
        #     return {"error": "BigQueryClient를 사용할 수 없습니다."}

        if preprocess_news_data is None:
            logger.error("[predict_market_impact] 뉴스 전처리 모듈을 사용할 수 없음")
            return {"error": "뉴스 전처리 모듈을 사용할 수 없습니다."}

        try:
            # bq = BigQueryClient()

            # 1. 데이터 가져오기 (뉴스 7일치, 가격 30일치)
            # news_df = bq.get_news_for_prediction(
            #     target_date,
            #     lookback_days=7,
            #     dataset_id="tilda",
            #     table_id="corn_all_news_with_sentiment",
            # )

            logger.debug("[predict_market_impact] BigQuery 서비스 획득 시작")
            bq_migrate = get_bq_service()
            logger.debug(
                f"[predict_market_impact] BigQuery 서비스 획득 성공: {type(bq_migrate).__name__}"
            )

            logger.info(
                f"[predict_market_impact] 뉴스 데이터 조회 중 - target_date: {target_date}, lookback_days: 7"
            )
            news_df = bq_migrate.get_news_articles_resources_features_corn(
                target_date=target_date,
                lookback_days=7,
            )
            logger.debug(
                f"[predict_market_impact] 뉴스 데이터 조회 완료 - 행 개수: {len(news_df)}, 열: {list(news_df.columns) if not news_df.empty else 'N/A'}"
            )

            logger.info(
                f"[predict_market_impact] 가격 데이터 조회 중 - commodity: corn, target_date: {target_date}, lookback_days: 30"
            )
            price_df = bq_migrate.get_price_history(
                commodity="corn",
                target_date=target_date,
                lookback_days=30,
            )
            logger.debug(
                f"[predict_market_impact] 가격 데이터 조회 완료 - 행 개수: {len(price_df)}, 열: {list(price_df.columns) if not price_df.empty else 'N/A'}"
            )

            if news_df.empty:
                logger.warning(
                    f"[predict_market_impact] 뉴스 데이터 없음 - target_date: {target_date}"
                )
                return {"error": f"{target_date} 기준 최근 뉴스 데이터가 없습니다."}
            if price_df.empty:
                logger.warning(
                    f"[predict_market_impact] 가격 데이터 없음 - target_date: {target_date}"
                )
                return {"error": f"{target_date} 기준 최근 가격 데이터가 없습니다."}

            logger.info(
                f"[predict_market_impact] 데이터 조회 성공 - 뉴스: {len(news_df)}행, 가격: {len(price_df)}행"
            )
            # 임베딩 컬럼 제외하고 로깅 (출력량 감소)
            exclude_cols = ["article_embedding", "embedding"]
            news_display_cols = [c for c in news_df.columns if c not in exclude_cols]
            logger.debug(
                f"[predict_market_impact] 뉴스 데이터 샘플 (처음 3행, 임베딩 제외):\n{news_df[news_display_cols].head(3).to_string()}"
            )
            logger.debug(
                f"[predict_market_impact] 가격 데이터 샘플 (처음 3행):\n{price_df.head(3).to_string()}"
            )

            # 2. 전처리 (문자열 임베딩 -> 배열 변환 등)
            logger.debug("[predict_market_impact] 뉴스 데이터 전처리 시작")
            processed_news = preprocess_news_data(news_df)
            logger.debug(
                f"[predict_market_impact] 뉴스 데이터 전처리 완료 - 결과 shape: {processed_news.shape if hasattr(processed_news, 'shape') else type(processed_news)}"
            )

            # 3. 예측 수행 (근거 뉴스 포함)
            # news_df_full에 원본 news_df를 넘겨서 BQ에서 가져온 데이터를 그대로 근거 추출에 사용
            logger.info(
                f"[predict_market_impact] 예측 시작 - target_date: {target_date}, top_k: 3"
            )
            result = self.predictor.predict_with_evidence(
                news_data=processed_news,
                price_history=price_df,
                target_date=target_date,
                news_df_full=news_df,  # 이미 로드한 데이터 재사용
                top_k=3,  # 상위 3개 근거 뉴스 추출
            )
            logger.info(
                f"[predict_market_impact] 예측 성공 - 결과 키: {list(result.keys()) if isinstance(result, dict) else type(result)}"
            )
            logger.debug(f"[predict_market_impact] 예측 결과 상세: {result}")

            return result

        except Exception as e:
            logger.exception(
                f"[predict_market_impact] 시장 예측 중 예기치 않은 오류 발생: {e}"
            )
            traceback.print_exc()
            return {"error": f"시장 예측 중 오류 발생: {str(e)}"}


if __name__ == "__main__":
    # 테스트 코드
    analyzer = SentimentAnalyzer()

    # 시장 예측 테스트 (BQ 연결 필요)
    print("\n시장 예측 테스트 (2026-01-27):")
    print(analyzer.predict_market_impact("2026-01-27"))
