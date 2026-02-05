import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import yaml
import os
import time
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path=None):
    """YAML 설정 파일 로드"""
    if config_path is None:
        # 기본 경로: 현재 파일과 같은 디렉토리의 config.yaml
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, 'config.yaml')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


class TimeSeriesXGBoostInference:
    """
    BigQuery에서 가져온 DataFrame을 사용하여 Walk-Forward 방식으로 학습하고 예측하는 클래스
    inference.py와 동일한 인터페이스를 제공합니다.
    """
    def __init__(self, config_path=None):
        """
        추론 엔진 초기화
        
        Args:
            config_path (str, optional): config.yaml 파일 경로. None이면 기본 경로 사용.
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = load_config(config_path)
        self.xgb_config = self.config['xgboost']
        self.validation_config = self.config['validation']
        
    def predict(self, history_df: pd.DataFrame, target_date: str) -> Dict[str, Any]:
        """
        제공된 과거 데이터를 사용하여 Walk-Forward 방식으로 학습하고 시장 방향을 예측합니다.
        
        Args:
            history_df (pd.DataFrame): Prophet 피처가 포함된 데이터프레임.
                                    'ds' 컬럼과 학습에 사용된 모든 피처가 포함되어야 합니다.
                                    target_date를 포함한 과거 데이터가 있어야 합니다.
            target_date (str): 예측할 날짜 문자열 ('YYYY-MM-DD' 형식).
            
        Returns:
            Dict: 예측 상세 결과 사전 (inference.py와 동일한 형식).
        """
        print(f"\n{'='*70}")
        print(f"🤖 XGBoost 시계열 예측 시작")
        print(f"{'='*70}")
        print(f"📅 타겟 날짜: {target_date}")
        
        try:
            target_ts = pd.Timestamp(target_date)
        except ValueError:
            raise ValueError(f"잘못된 날짜 형식입니다: {target_date}. YYYY-MM-DD 형식을 사용하세요.")
        
        # 'ds' 컬럼이 datetime 형식인지 확인
        if not pd.api.types.is_datetime64_any_dtype(history_df['ds']):
            history_df['ds'] = pd.to_datetime(history_df['ds'])
        
        # 타겟 날짜에 해당하는 행 찾기
        target_idx = history_df[history_df['ds'] == target_ts].index
        
        if len(target_idx) == 0:
            raise ValueError(f"제공된 데이터프레임에서 해당 날짜({target_date})의 데이터를 찾을 수 없습니다.")
        
        target_idx = target_idx[0]
        target_row_idx = history_df.index.get_loc(target_idx)
        
        # 피처 컬럼 정의
        exclude_cols = ['ds', 'y', 'direction', 'y_change', 'yhat_lower', 'yhat_upper', 'EMA', 'Volume']
        feature_columns = [col for col in history_df.columns if col not in exclude_cols]
        
        print(f"\n📊 데이터 준비")
        print(f"  - 전체 데이터 크기: {len(history_df)} 행")
        print(f"  - 사용 피처 수: {len(feature_columns)} 개")
        
        # Walk-Forward 학습을 위한 데이터 준비
        min_train_samples = self.validation_config['min_train_samples']
        window_size = self.validation_config.get('window_size', None)
        train_val_split = self.xgb_config['train_val_split']
        
        # 타겟 날짜까지의 데이터만 사용 (미래 데이터는 사용하지 않음)
        df_until_target = history_df.iloc[:target_row_idx + 1].copy()
        
        if len(df_until_target) < min_train_samples:
            raise ValueError(
                f"학습에 필요한 최소 샘플 수({min_train_samples})보다 적습니다. "
                f"현재 사용 가능한 샘플 수: {len(df_until_target)}"
            )
        
        # Sliding Window 적용
        if window_size is None:
            train_val_start = 0
        else:
            train_val_start = max(0, target_row_idx - window_size)
        
        available_samples = target_row_idx - train_val_start
        train_size_relative = int(available_samples * train_val_split)
        train_end = train_val_start + train_size_relative
        
        # Train/Val/Test 분리
        X_train = df_until_target.iloc[train_val_start:train_end][feature_columns]
        y_train = df_until_target.iloc[train_val_start:train_end]['direction']
        
        X_val = df_until_target.iloc[train_end:target_row_idx][feature_columns]
        y_val = df_until_target.iloc[train_end:target_row_idx]['direction']
        
        X_test = df_until_target.iloc[target_row_idx:target_row_idx+1][feature_columns]
        row = df_until_target.iloc[target_row_idx:target_row_idx+1]
        
        print(f"\n📂 데이터 분할 완료")
        print(f"  - Train 데이터: {len(X_train)} 행")
        print(f"  - Val 데이터: {len(X_val)} 행")
        print(f"  - Test 데이터: {len(X_test)} 행")
        
        # 클래스 불균형 처리
        n_positive = (y_train == 1).sum()
        n_negative = (y_train == 0).sum()
        scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1
        
        print(f"\n⚖️  클래스 분포")
        print(f"  - 상승(1): {n_positive} 개 ({n_positive/len(y_train)*100:.1f}%)")
        print(f"  - 하락(0): {n_negative} 개 ({n_negative/len(y_train)*100:.1f}%)")
        print(f"  - Scale Pos Weight: {scale_pos_weight:.3f}")
        
        # XGBoost 파라미터 설정
        xgb_params = {
            'objective': self.xgb_config['objective'],
            'max_depth': self.xgb_config['max_depth'],
            'learning_rate': self.xgb_config['learning_rate'],
            'n_estimators': self.xgb_config['n_estimators'],
            'min_child_weight': self.xgb_config['min_child_weight'],
            'subsample': self.xgb_config['subsample'],
            'colsample_bytree': self.xgb_config['colsample_bytree'],
            'gamma': self.xgb_config['gamma'],
            'reg_alpha': self.xgb_config['reg_alpha'],
            'reg_lambda': self.xgb_config['reg_lambda'],
            'scale_pos_weight': scale_pos_weight,
            'random_state': self.xgb_config['random_state'],
            'verbosity': 0
        }
        
        # 모델 학습
        early_stopping_rounds = self.xgb_config.get('early_stopping_rounds')
        
        print(f"\n🚀 XGBoost 모델 학습 시작")
        print(f"  - Max Depth: {xgb_params['max_depth']}")
        print(f"  - Learning Rate: {xgb_params['learning_rate']}")
        print(f"  - N Estimators: {xgb_params['n_estimators']}")
        
        train_start_time = time.time()
        
        if len(X_val) > 0 and early_stopping_rounds is not None:
            print(f"  - Early Stopping: {early_stopping_rounds} rounds")
            xgb_params['early_stopping_rounds'] = early_stopping_rounds
            xgb_model = XGBClassifier(**xgb_params)
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
        else:
            print(f"  - Early Stopping: 미사용")
            xgb_model = XGBClassifier(**xgb_params)
            xgb_model.fit(X_train, y_train)
        
        train_time = time.time() - train_start_time
        print(f"✅ 모델 학습 완료 (소요 시간: {train_time:.2f}초)")
        
        # 예측 수행
        print(f"\n🎯 예측 수행 중...")
        prediction_prob = xgb_model.predict_proba(X_test)[0]  # [하락확률, 상승확률]
        prediction = xgb_model.predict(X_test)[0]  # 0 또는 1
        
        # 예측 방향에 따른 신뢰도(확률) 추출
        confidence = prediction_prob[1] if prediction == 1 else prediction_prob[0]
        
        # BigQuery에서 가져온 Prophet features를 반환할 딕셔너리 준비
        result = {
            "target_date": target_date,
            "forecast_direction": "Up" if prediction == 1 else "Down",
            "confidence_score": float(confidence) * 100  # 신뢰도 (%) 추가
        }
        
        # Prophet feature들을 결과에 추가
        # row의 모든 컬럼 값을 결과에 포함
        for col in row.columns:
            if col not in ['ds']:  # ds는 이미 target_date로 포함됨
                value = row[col].values[0]
                # NaN 체크 후 변환
                if pd.isna(value):
                    result[col] = None
                elif isinstance(value, (bool, np.bool_)):
                    # bool 타입은 int로 변환 (JSON 직렬화 호환)
                    result[col] = int(value)
                elif isinstance(value, (np.integer, np.floating, int, float)):
                    result[col] = float(value)
                else:
                    result[col] = value
        
        print(f"\n📈 예측 결과")
        print(f"  - xgboost 예측 방향: {result['forecast_direction']} ({'상승' if prediction == 1 else '하락'})")
        if 'yhat' in result and result['yhat'] is not None:
            print(f"  - Prophet 예측값: {result['yhat']:.2f}")
        if 'y' in result and result['y'] is not None:
            print(f"  - 어제 종가: {result['y']:.2f}")
        print(f"{'='*70}\n")
        
        return result
