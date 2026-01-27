import numpy as np
from typing import List, Tuple


class TimeSeriesPredictor:
    """시계열 예측 모델 (테스트용 틀)"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.scaler_min = None
        self.scaler_max = None
    
    def predict(self, data: List[float], lookback: int = 60) -> Tuple[float, float]:
        """시계열 데이터 예측"""
        # 테스트용 더미 구현
        if not data:
            return 100.0, 0.5
        
        # 최근 값들의 평균을 기반으로 예측
        recent_values = data[-10:] if len(data) >= 10 else data
        prediction = sum(recent_values) / len(recent_values)
        
        # 신뢰도는 데이터 길이와 변동성 기반
        if len(data) >= 10:
            variance = np.var(recent_values)
            confidence = max(0.5, min(0.95, 1.0 - variance / 100.0))
        else:
            confidence = 0.6
        
        return float(prediction), float(confidence)
    
    def train(self, data: List[float], epochs: int = 50, batch_size: int = 32, lookback: int = 60):
        """모델 학습"""
        pass
    
    def load_model_weights(self):
        """학습된 모델 가중치 로드"""
        pass
