"""
뉴스 감성 분석 모듈

BigQuery에서 뉴스 및 가격 데이터를 가져와 시장 영향력을 예측합니다.
"""

import logging
from typing import Dict, Any

from libs.gcp import GCPServiceFactory
from libs.utils.config import get_config

logger = logging.getLogger(__name__)


# 뉴스 모델 컴포넌트 임포트 (지연 로딩)
try:
    from model.news_sentiment_model.inference_with_evidence import CornPricePredictor
    from model.news_sentiment_model.preprocessing import preprocess_news_data
except ImportError as e:
    logger.warning(f"뉴스 감성 모델 모듈 임포트 실패: {e}")
    CornPricePredictor = None
    preprocess_news_data = None


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

    def __init__(self):
        """SentimentAnalyzer 초기화"""
        self.predictor = None
        self._bq = None
        self._initialize()

    def _initialize(self):
        """모델 및 서비스 초기화"""
        # BigQueryService 초기화
        config = get_config()
        factory = GCPServiceFactory()
        self._bq = factory.get_bigquery_client(dataset_id=config.bigquery.dataset_id)
        logger.debug("BigQueryService initialized")

        # 예측 모델 초기화
        try:
            if CornPricePredictor is not None:
                self.predictor = CornPricePredictor()
                self.predictor.load_model()
                logger.debug("CornPricePredictor initialized")
        except Exception as e:
            logger.error(f"모델 초기화 중 오류 발생: {e}")

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
        if self.predictor is None:
            return {"error": "뉴스 예측 모델이 로드되지 않았습니다."}

        if preprocess_news_data is None:
            return {"error": "뉴스 전처리 모듈을 사용할 수 없습니다."}

        try:
            # 1. 데이터 가져오기 (뉴스 7일치, 가격 30일치)
            news_df = self._bq.get_news_for_prediction(
                target_date=target_date,
                lookback_days=7,
            )
            price_df = self._bq.get_price_history(
                commodity="corn",
                target_date=target_date,
                lookback_days=30,
            )

            if news_df.empty:
                return {"error": f"{target_date} 기준 최근 뉴스 데이터가 없습니다."}
            if price_df.empty:
                return {"error": f"{target_date} 기준 최근 가격 데이터가 없습니다."}

            logger.info(f"Retrieved {len(news_df)} news, {len(price_df)} prices for {target_date}")

            # 2. 전처리 (문자열 임베딩 -> 배열 변환 등)
            processed_news = preprocess_news_data(news_df)

            # 3. 예측 수행 (근거 뉴스 포함)
            # news_df_full에 원본 news_df를 넘겨서 BQ에서 가져온 데이터를 그대로 근거 추출에 사용
            result = self.predictor.predict_with_evidence(
                news_data=processed_news,
                price_history=price_df,
                target_date=target_date,
                news_df_full=news_df,  # 이미 로드한 데이터 재사용
                top_k=3,  # 상위 3개 근거 뉴스 추출
            )

            return result

        except Exception as e:
            logger.exception("시장 예측 중 오류 발생")
            return {"error": f"시장 예측 중 오류 발생: {str(e)}"}


if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    analyzer = SentimentAnalyzer()

    # 시장 예측 테스트 (BQ 연결 필요)
    print("\n시장 예측 테스트 (2026-01-27):")
    print(analyzer.predict_market_impact("2026-01-27"))
