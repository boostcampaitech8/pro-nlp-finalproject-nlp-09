"""
옥수수 선물 가격 예측 모델 추론 코드
저장된 모델을 로드하여 새로운 데이터에 대한 예측을 수행합니다.
"""

import pandas as pd
import numpy as np
import json
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

import xgboost as xgb

# 공통 전처리 모듈
from preprocessing import prepare_inference_features


class CornPricePredictor:
    """옥수수 가격 예측 모델 클래스"""
    
    def __init__(self, model_dir='models'):
        """
        모델 초기화
        
        Args:
            model_dir: 모델 파일들이 저장된 디렉토리
        """
        self.model_dir = model_dir
        self.model = None
        self.pca = None
        self.feature_columns = None
        
    def load_model(self):
        """저장된 모델, PCA 객체, 피처 컬럼 로드"""
        print("=" * 80)
        print("모델 로딩 중...")
        print("=" * 80)
        
        # 1. XGBoost 모델 로드
        model_path = os.path.join(self.model_dir, 'xgb_model.json')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        print(f"✓ 모델 로드 완료: {model_path}")
        
        # 2. PCA 객체 로드
        pca_path = os.path.join(self.model_dir, 'pca_transformer.pkl')
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"PCA 파일을 찾을 수 없습니다: {pca_path}")
        
        with open(pca_path, 'rb') as f:
            self.pca = pickle.load(f)
        print(f"✓ PCA 객체 로드 완료: {pca_path}")
        
        # 3. 피처 컬럼 로드
        feature_path = os.path.join(self.model_dir, 'feature_columns.json')
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"피처 컬럼 파일을 찾을 수 없습니다: {feature_path}")
        
        with open(feature_path, 'r') as f:
            self.feature_columns = json.load(f)
        print(f"✓ 피처 컬럼 로드 완료: {feature_path}")
        print(f"  - 총 피처 개수: {len(self.feature_columns)}")
        
        print("\n모든 모델 구성요소가 성공적으로 로드되었습니다!")
        print("=" * 80)
        
    def predict_next_day(self, news_data, price_history):
        """
        다음 날 가격 상승 여부 예측
        
        Args:
            news_data: 최근 뉴스 데이터 (DataFrame)
                필수 컬럼: ['publish_date', 'article_embedding', 'price_impact_score', 
                        'sentiment_confidence', 'positive_score', 'negative_score']
                최소 3일치 데이터 권장
            
            price_history: 최근 가격 데이터 (DataFrame)
                필수 컬럼: ['time' 또는 'date', 'close', 'ret_1d']
                최소 5일치 데이터 권장
        
        Returns:
            dict: 예측 결과
                {
                    "prediction": 0 or 1,  # 0: 하락, 1: 상승
                    "probability": float,   # 상승 확률 (0~1)
                    "features_summary": {
                        "latest_news_count": int,
                        "avg_sentiment": float,
                        "latest_price": float,
                        "data_points_used": int
                    }
                }
        """
        # 모델 로드 확인
        if self.model is None or self.pca is None or self.feature_columns is None:
            raise RuntimeError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        print("\n" + "=" * 80)
        print("예측 데이터 전처리 중...")
        print("=" * 80)
        
        # 입력 데이터 검증
        self._validate_input_data(news_data, price_history)
        
        # 전처리 및 피처 생성
        try:
            X = prepare_inference_features(
                news_data=news_data,
                price_history=price_history,
                pca_transformer=self.pca,
                feature_columns=self.feature_columns
            )
            print(f"✓ 전처리 완료: 피처 shape = {X.shape}")
        except Exception as e:
            raise ValueError(f"데이터 전처리 중 오류 발생: {str(e)}")
        
        # 예측 수행
        print("\n예측 수행 중...")
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0, 1]  # 상승(1) 확률
        
        # 결과 요약 생성
        features_summary = self._create_features_summary(news_data, price_history)
        
        # 결과 딕셔너리 생성
        result = {
            "prediction": int(prediction),
            "probability": float(probability),
            "features_summary": features_summary
        }
        
        print("=" * 80)
        print("예측 완료!")
        print("=" * 80)
        print(f"예측 결과: {'상승(1)' if prediction == 1 else '하락(0)'}")
        print(f"상승 확률: {probability:.2%}")
        print("=" * 80)
        
        return result
    
    def _validate_input_data(self, news_data, price_history):
        """입력 데이터 검증"""
        # 뉴스 데이터 검증
        required_news_cols = ['publish_date', 'article_embedding', 'price_impact_score', 
                            'sentiment_confidence', 'positive_score', 'negative_score']
        missing_news_cols = [col for col in required_news_cols if col not in news_data.columns]
        if missing_news_cols:
            raise ValueError(f"뉴스 데이터에 필수 컬럼이 누락되었습니다: {missing_news_cols}")
        
        # 가격 데이터 검증
        if 'time' not in price_history.columns and 'date' not in price_history.columns:
            raise ValueError("가격 데이터에 'time' 또는 'date' 컬럼이 필요합니다.")
        
        required_price_cols = ['close', 'ret_1d']
        price_history['ret_1d'] = np.log(price_history['close'] / price_history['close'].shift(1))
        missing_price_cols = [col for col in required_price_cols if col not in price_history.columns]
        if missing_price_cols:
            raise ValueError(f"가격 데이터에 필수 컬럼이 누락되었습니다: {missing_price_cols}")
        
        # 데이터 개수 확인
        if len(news_data) < 3:
            print(f"경고: 뉴스 데이터가 {len(news_data)}개로 부족합니다. 최소 3일치 권장")
        
        if len(price_history) < 5:
            print(f"경고: 가격 데이터가 {len(price_history)}개로 부족합니다. 최소 5일치 권장")
    
    def _create_features_summary(self, news_data, price_history):
        """피처 요약 정보 생성"""
        summary = {
            "latest_news_count": int(len(news_data)),
            "avg_sentiment": float(news_data['sentiment_confidence'].mean()),
            "avg_price_impact": float(news_data['price_impact_score'].mean()),
            "latest_price": float(price_history['close'].iloc[-1]),
            "data_points_used": int(len(price_history))
        }
        return summary


def predict_next_day(news_data, price_history, model_dir='models'):
    """
    편의 함수: 다음 날 가격 상승 여부 예측
    
    Args:
        news_data: 최근 뉴스 데이터 (DataFrame)
        price_history: 최근 가격 데이터 (DataFrame)
        model_dir: 모델 파일들이 저장된 디렉토리
    
    Returns:
        dict: 예측 결과
    """
    predictor = CornPricePredictor(model_dir=model_dir)
    predictor.load_model()
    result = predictor.predict_next_day(news_data, price_history)
    return result


# ============================================
# 사용 예시
# ============================================
if __name__ == '__main__':
    """
    추론 코드 사용 예시
    """
    print("\n" + "=" * 80)
    print("추론 코드 사용 예시")
    print("=" * 80)
    
    # 예시 데이터 생성 (실제로는 최근 데이터를 로드해야 함)
    print("\n※ 실제 사용 시에는 아래처럼 최근 데이터를 로드하세요:")
    print("""
    # 최근 뉴스 데이터 로드 (최소 3일치)
    news_data = pd.read_csv('recent_news.csv')
    
    # 최근 가격 데이터 로드 (최소 5일치)
    price_history = pd.read_csv('recent_prices.csv')
    
    # 예측 수행
    result = predict_next_day(news_data, price_history, model_dir='models')
    
    # 결과 확인
    print(f"예측: {result['prediction']}")  # 0: 하락, 1: 상승
    print(f"상승 확률: {result['probability']:.2%}")
    print(f"피처 요약: {result['features_summary']}")
    """)
    
    print("\n" + "=" * 80)
    print("LangChain 연동 예시")
    print("=" * 80)
    print("""
    # LangChain과 연동하여 보고서 생성
    from langchain.tools import Tool
    
    def corn_price_prediction_tool(input_data):
        \"\"\"옥수수 가격 예측 도구\"\"\"
        news_data = load_recent_news()  # 최근 뉴스 로드
        price_history = load_recent_prices()  # 최근 가격 로드
        
        result = predict_next_day(news_data, price_history)
        
        # LLM에게 전달할 텍스트 생성
        report = f\"\"\"
        옥수수 선물 가격 예측 결과:
        - 예측: {'상승' if result['prediction'] == 1 else '하락'}
        - 상승 확률: {result['probability']:.2%}
        - 분석 데이터:
          * 뉴스 기사 수: {result['features_summary']['latest_news_count']}개
          * 평균 감성 점수: {result['features_summary']['avg_sentiment']:.2f}
          * 평균 가격 영향 점수: {result['features_summary']['avg_price_impact']:.2f}
          * 최근 가격: ${result['features_summary']['latest_price']:.2f}
        \"\"\"
        return report
    
    # LangChain Tool로 등록
    prediction_tool = Tool(
        name="CornPricePrediction",
        func=corn_price_prediction_tool,
        description="옥수수 선물 가격의 다음날 상승/하락을 예측합니다."
    )
    
    # LLM 에이전트에 도구 추가
    agent = initialize_agent(
        tools=[prediction_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )
    
    # 보고서 생성 요청
    response = agent.run("옥수수 가격 전망 보고서를 작성해주세요.")
    print(response)
    """)
    
    print("\n" + "=" * 80)
