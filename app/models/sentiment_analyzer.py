from typing import Dict, Any
import traceback

# 1. 뉴스 모델 컴포넌트 임포트
try:
    from model.news_sentiment_model.inference_with_evidence import CornPricePredictor
    from model.news_sentiment_model.preprocessing import preprocess_news_data
except ImportError as e:
    print(f"경고: 뉴스 감성 모델 모듈 임포트 실패: {e}")
    CornPricePredictor = None

# 2. BigQuery 클라이언트 임포트
try:
    from app.utils.bigquery_client import BigQueryClient
except ImportError as e:
    print(f"경고: BigQueryClient 임포트 실패: {e}")
    BigQueryClient = None


class SentimentAnalyzer:
    """
    뉴스 감성 분석 및 시장 영향력 예측기 (Adapter)
    """

    def __init__(self):
        self.predictor = None
        self._initialize_models()

    def _initialize_models(self):
        """모델 지연 초기화"""
        try:
            if CornPricePredictor:
                self.predictor = CornPricePredictor()
                self.predictor.load_model()
        except Exception as e:
            print(f"모델 초기화 중 오류 발생: {e}")

    def predict_market_impact(self, target_date: str, commodity: str = "corn") -> Dict[str, Any]:
        """
        특정 날짜의 뉴스 기반 시장 예측 (근거 뉴스 포함)

        Args:
            target_date: 분석할 날짜 (YYYY-MM-DD)
            commodity: 상품명 (corn, soybean, wheat)

        Returns:
            Dict: 예측 결과 및 근거 뉴스 리스트
        """
        if not self.predictor:
            return {"error": "뉴스 예측 모델이 로드되지 않았습니다."}

        if not BigQueryClient:
            return {"error": "BigQueryClient를 사용할 수 없습니다."}

        try:
            bq = BigQueryClient()

            # 1. 데이터 가져오기 (품목별 필터링 적용)
            news_df = bq.get_news_for_prediction(target_date, lookback_days=7, commodity=commodity)
            price_df = bq.get_price_history(target_date, lookback_days=30, commodity=commodity)

            if news_df.empty:
                return {"error": f"{commodity} - {target_date} 기준 최근 뉴스 데이터가 없습니다."}
            if price_df.empty:
                return {"error": f"{commodity} - {target_date} 기준 최근 가격 데이터가 없습니다."}

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
            traceback.print_exc()
            return {"error": f"시장 예측 중 오류 발생: {str(e)}"}


if __name__ == "__main__":
    # 테스트 코드
    analyzer = SentimentAnalyzer()

    # 시장 예측 테스트 (BQ 연결 필요)
    print("\n시장 예측 테스트 (2026-01-27):")
    print(analyzer.predict_market_impact("2026-01-27"))
