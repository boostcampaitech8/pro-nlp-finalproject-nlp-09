import pandas as pd
import pickle
import os
from typing import Dict, Any

class TimeSeriesInference:
    def __init__(self, model_path: str = None):
        """
        추론 엔진 초기화 (모델 로드 전용).
        """
        # 현재 파일의 기본 경로
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 모델 경로 설정
        # 우선순위: 1. 인자값 2. 환경변수 3. 기본 로컬 경로
        self.model_path = (
            model_path 
            or os.getenv('TS_MODEL_PATH') 
            or os.path.join(self.base_dir, 'xgboost_model.pkl')
        )
            
        self.model = None
        self._load_model()

    def _load_model(self):
        """로컬에서 XGBoost 모델을 로드합니다."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다: {self.model_path}.\n"
                "힌트: 'TS_MODEL_PATH' 환경변수를 설정하거나 파일이 로컬에 존재하는지 확인하세요."
            )
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"모델 로드 실패 ({self.model_path}): {e}")

    def predict(self, history_df: pd.DataFrame, target_date: str) -> Dict[str, Any]:
        """
        제공된 과거 데이터를 사용하여 시장 방향을 예측합니다.
        
        Args:
            history_df (pd.DataFrame): Prophet 피처가 포함된 데이터프레임.
                                    'ds' 컬럼과 학습에 사용된 모든 피처가 포함되어야 합니다.
                                    target_date를 포함한 과거 데이터가 있어야 합니다.
            target_date (str): 예측할 날짜 문자열 ('YYYY-MM-DD' 형식).
            
        Returns:
            Dict: 예측 상세 결과 사전.
        """
        try:
            target_ts = pd.Timestamp(target_date)
        except ValueError:
            raise ValueError(f"잘못된 날짜 형식입니다: {target_date}. YYYY-MM-DD 형식을 사용하세요.")
        
        # 'ds' 컬럼이 datetime 형식인지 확인
        if not pd.api.types.is_datetime64_any_dtype(history_df['ds']):
            history_df['ds'] = pd.to_datetime(history_df['ds'])

        # 타겟 날짜에 해당하는 행 찾기
        row = history_df[history_df['ds'] == target_ts]
        
        if row.empty:
            raise ValueError(f"제공된 데이터프레임에서 해당 날짜({target_date})의 데이터를 찾을 수 없습니다.")

        # 피처 컬럼 정의 (학습 시 제외했던 컬럼들 제거)
        exclude_cols = ['ds', 'y', 'direction', 'y_change', 'yhat_lower', 'yhat_upper']
        feature_cols = [col for col in history_df.columns if col not in exclude_cols]

        # XGBoost용 피처 추출
        X_test = row[feature_cols]
        
        # XGBoost 예측 (방향)
        prediction_prob = self.model.predict_proba(X_test)[0] # [하락확률, 상승확률]
        prediction = self.model.predict(X_test)[0] # 0 또는 1
        
        confidence = prediction_prob[1] if prediction == 1 else prediction_prob[0]
        
        # Prophet 예측값 (yhat)
        yhat = row['yhat'].values[0]
        
        # 문맥 통계 (추세 분석용)
        # 제공된 과거 데이터에서 최근 7일 평균 계산
        recent_7_days = history_df.tail(7)
        recent_mean = recent_7_days['yhat'].mean()
        
        # 전 기간 평균 계산 (주의: 전달된 DataFrame의 기간에 따라 다름)
        all_time_mean = history_df['yhat'].mean()
        
        # 보고서에 필요한 형태로 결과 반환
        return {
            "target_date": target_date,
            "forecast_value": float(yhat),         # Prophet 예측값
            "forecast_direction": "Up" if prediction == 1 else "Down",
            "confidence_score": float(confidence) * 100, # 신뢰도 (%)
            "recent_mean_7d": float(recent_mean),  # 최근 7일 평균
            "all_time_mean": float(all_time_mean), # 전체 기간 평균
            "trend_analysis": "Rising" if yhat > recent_mean else "Falling", # 단순 추세
            "volatility_index": float(recent_7_days['yhat'].std()), # 변동성 지표 (표준편차)
            "last_observed_value": float(row['y'].values[0]) if 'y' in row and not pd.isna(row['y'].values[0]) else None # 실제값 (있으면)
        }

if __name__ == "__main__":
    try:
        inference = TimeSeriesInference()
        test_date = "2026-01-27" 
        result = inference.predict(test_date)
        print(f"Prediction for {test_date}:")
        import json
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
