"""
Prophet Feature 추출 - Wheat (BigQuery 버전)
=============================================
BigQuery에서 Wheat 데이터를 읽어서 Prophet으로 예측하고,
생성된 features를 CSV로 저장합니다.

Granger Causality 검증 결과:
- Wheat Close ← Corn Close (lag 2)
- Wheat Close ← Soybean Close (lag 1)
- Wheat Close ← Wheat EMA (lag 1)
- Wheat Close ← Wheat Volume (lag 1)
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import yaml
from tqdm import tqdm
import warnings
import random
from google.cloud import bigquery
from datetime import datetime
import os
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# .env 파일 로드
load_dotenv()

# 재현성을 위한 시드 고정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_config(config_path="config.yaml"):
    """YAML 설정 파일 로드"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_data_from_bigquery(project_id, dataset_id, wheat_table_id, corn_table_id=None, soybean_table_id=None, end_date="2026-02-03"):
    """
    BigQuery에서 데이터 로드 (Granger 검증 결과 반영)
    Wheat 데이터 + Corn 데이터 + Soybean 데이터 (선택적)
    """
    print(f"📊 BigQuery에서 데이터 로딩 중...")
    print(f"   Project: {project_id}")
    print(f"   Dataset: {dataset_id}")
    print(f"   Wheat Table: {wheat_table_id}")
    if corn_table_id:
        print(f"   Corn Table: {corn_table_id}")
    if soybean_table_id:
        print(f"   Soybean Table: {soybean_table_id}")
    print(f"   End Date: {end_date}")
    
    client = bigquery.Client(project=project_id)
    
    # Wheat 데이터 로드
    query_wheat = f"""
    SELECT 
        time,
        open,
        high,
        low,
        close,
        ema as EMA,
        volume as Volume
    FROM `{project_id}.{dataset_id}.{wheat_table_id}`
    WHERE DATE(time) <= '{end_date}'
    ORDER BY time
    """
    
    try:
        df_wheat = client.query(query_wheat).to_dataframe()
    except Exception as e:
        # 컬럼명이 대문자일 경우를 대비한 재시도
        print(f"   첫 번째 쿼리 실패, 대체 쿼리 시도...")
        query_wheat = f"""
        SELECT 
            time,
            open,
            high,
            low,
            close,
            EMA,
            Volume
        FROM `{project_id}.{dataset_id}.{wheat_table_id}`
        WHERE DATE(time) <= '{end_date}'
        ORDER BY time
        """
        df_wheat = client.query(query_wheat).to_dataframe()
    
    print(f"✅ Wheat 데이터 로드 완료: {len(df_wheat)} 행")
    print(f"   기간: {df_wheat['time'].min()} ~ {df_wheat['time'].max()}")
    
    df = df_wheat.copy()
    
    # Corn 데이터 로드 (Granger 검증 결과: lag2 영향)
    if corn_table_id:
        print(f"\n📊 Corn 데이터 로딩 중...")
        query_corn = f"""
        SELECT 
            time,
            close as corn_close
        FROM `{project_id}.{dataset_id}.{corn_table_id}`
        WHERE DATE(time) <= '{end_date}'
        ORDER BY time
        """
        
        try:
            df_corn = client.query(query_corn).to_dataframe()
            print(f"✅ Corn 데이터 로드 완료: {len(df_corn)} 행")
            print(f"   기간: {df_corn['time'].min()} ~ {df_corn['time'].max()}")
            
            # 데이터 병합 (Wheat 기준 left join - 결측치 그대로 유지)
            df = pd.merge(df, df_corn, on='time', how='left')
            
            missing_count = df['corn_close'].isna().sum()
            print(f"   - Corn 데이터 병합: {len(df) - missing_count}일 사용 가능")
        except Exception as e:
            print(f"⚠️  Corn 데이터 로드 실패: {e}")
            print(f"   Corn 없이 진행합니다.")
    
    # Soybean 데이터 로드 (Granger 검증 결과: lag1 영향)
    if soybean_table_id:
        print(f"\n📊 Soybean 데이터 로딩 중...")
        query_soybean = f"""
        SELECT 
            time,
            close as soybean_close
        FROM `{project_id}.{dataset_id}.{soybean_table_id}`
        WHERE DATE(time) <= '{end_date}'
        ORDER BY time
        """
        
        try:
            df_soybean = client.query(query_soybean).to_dataframe()
            print(f"✅ Soybean 데이터 로드 완료: {len(df_soybean)} 행")
            print(f"   기간: {df_soybean['time'].min()} ~ {df_soybean['time'].max()}")
            
            # 데이터 병합 (Wheat 기준 left join - 결측치 그대로 유지)
            df = pd.merge(df, df_soybean, on='time', how='left')
            
            missing_count = df['soybean_close'].isna().sum()
            print(f"   - Soybean 데이터 병합: {len(df) - missing_count}일 사용 가능")
        except Exception as e:
            print(f"⚠️  Soybean 데이터 로드 실패: {e}")
            print(f"   Soybean 없이 진행합니다.")
    
    print(f"\n✅ 데이터 병합 완료: {len(df)} 행 (Wheat 기준)\n")
    
    return df


def preprocess_data(df):
    """데이터 전처리 (BigQuery 데이터 타입 변환 포함)"""
    df["ds"] = pd.to_datetime(df["time"])
    df["y"] = pd.to_numeric(df["close"], errors='coerce')
    
    # 숫자 컬럼들을 명시적으로 float로 변환 (BigQuery timestamp 호환성)
    df["Volume"] = pd.to_numeric(df["Volume"], errors='coerce')
    df["EMA"] = pd.to_numeric(df["EMA"], errors='coerce')
    
    # 선택 컬럼 정의
    cols = ["ds", "y", "Volume", "EMA"]
    
    # Corn이 있는 경우 포함
    if 'corn_close' in df.columns:
        df["corn_close"] = pd.to_numeric(df["corn_close"], errors='coerce')
        cols.append("corn_close")
    
    # Soybean이 있는 경우 포함
    if 'soybean_close' in df.columns:
        df["soybean_close"] = pd.to_numeric(df["soybean_close"], errors='coerce')
        cols.append("soybean_close")
    
    df = df[cols].copy()
    df = df.sort_values("ds").reset_index(drop=True)
    
    # 필수 컬럼(ds, y, Volume, EMA)만 NaN 체크
    # corn_close, soybean_close는 NaN이어도 괜찮음 (나중에 조건부로 사용)
    df = df.dropna(subset=['ds', 'y', 'Volume', 'EMA']).reset_index(drop=True)
    
    print(f"✅ 데이터 전처리 완료: {len(df)} 행")
    if 'corn_close' in df.columns:
        corn_available = df['corn_close'].notna().sum()
        print(f"   - Corn 데이터 있음: {corn_available}일")
    if 'soybean_close' in df.columns:
        soybean_available = df['soybean_close'].notna().sum()
        print(f"   - Soybean 데이터 있음: {soybean_available}일")
    print()
    
    return df


def create_granger_lag_features(df):
    """
    Granger 검증 결과 기반 Lag Features 생성
    - corn_close: lag 2 (있는 경우에만)
    - soybean_close: lag 1 (있는 경우에만)
    - EMA: lag 1 (필수)
    - Volume: lag 1 (필수)
    """
    df = df.copy()
    
    print("🔬 Granger 검증 결과 기반 Lag Features 생성 중...")
    print("  - Corn Close: lag 2 (데이터 있는 경우)")
    print("  - Soybean Close: lag 1 (데이터 있는 경우)")
    print("  - Wheat EMA: lag 1 (필수)")
    print("  - Wheat Volume: lag 1 (필수)")
    
    # Granger 검증 결과에 따른 특정 lag만 생성
    if 'corn_close' in df.columns:
        df['corn_close_lag2'] = df['corn_close'].shift(2).astype(float)
    
    if 'soybean_close' in df.columns:
        df['soybean_close_lag1'] = df['soybean_close'].shift(1).astype(float)
    
    df['EMA_lag1'] = df['EMA'].shift(1).astype(float)
    df['Volume_lag1'] = df['Volume'].shift(1).astype(float)
    
    # 필수 컬럼(EMA_lag1, Volume_lag1)만 NaN 체크
    df = df.dropna(subset=['EMA_lag1', 'Volume_lag1']).reset_index(drop=True)
    
    # Corn, Soybean lag는 NaN이어도 괜찮음
    print(f"\n✅ Lag Features 생성 완료: {len(df)} 행")
    if 'corn_close_lag2' in df.columns:
        corn_available = df['corn_close_lag2'].notna().sum()
        print(f"   - Corn lag2 사용 가능: {corn_available}일")
    if 'soybean_close_lag1' in df.columns:
        soybean_available = df['soybean_close_lag1'].notna().sum()
        print(f"   - Soybean lag1 사용 가능: {soybean_available}일")
    print()
    
    return df


def extract_prophet_features_walkforward(df, config):
    """
    Walk-Forward 방식으로 Prophet Features 추출 (Granger 검증 결과 반영)
    Corn, Soybean 데이터가 없는 경우 EMA, Volume만으로 예측
    """
    prophet_config = config["prophet"]
    train_window_days = int(prophet_config["train_window_years"] * 365)
    
    # 기본 regressors (항상 사용)
    base_regressors = ['EMA_lag1', 'Volume_lag1']
    has_corn = 'corn_close_lag2' in df.columns
    has_soybean = 'soybean_close_lag1' in df.columns
    
    print("🔮 Walk-Forward Prophet Feature 추출 시작 (Granger 검증 기반)")
    print(f"   학습 윈도우: {prophet_config['train_window_years']}년 ({train_window_days}일)")
    print(f"   기본 Regressors: {base_regressors}")
    if has_corn:
        print(f"   조건부 Regressor: corn_close_lag2 (데이터 있는 경우에만 사용)")
    if has_soybean:
        print(f"   조건부 Regressor: soybean_close_lag1 (데이터 있는 경우에만 사용)")
    print()
    
    prophet_features_list = []
    start_idx = train_window_days
    
    with tqdm(total=len(df) - start_idx - 1, desc="Prophet 학습 및 예측") as pbar:
        for i in range(start_idx, len(df) - 1):
            train_start_idx = max(0, i - train_window_days)
            train_end_idx = i
            train_subset = df.iloc[train_start_idx:train_end_idx].copy()
            test_subset = df.iloc[i + 1 : i + 2].copy()
            
            # 이번 예측에 사용할 regressors 결정
            regressors = base_regressors.copy()
            
            # Corn 데이터 확인
            use_corn = False
            if has_corn and pd.notna(test_subset['corn_close_lag2'].values[0]):
                corn_ratio = train_subset['corn_close_lag2'].notna().sum() / len(train_subset)
                if corn_ratio > 0.5:  # 학습 데이터의 50% 이상에 Corn이 있어야 사용
                    use_corn = True
                    regressors.append('corn_close_lag2')
            
            # Soybean 데이터 확인
            use_soybean = False
            if has_soybean and pd.notna(test_subset['soybean_close_lag1'].values[0]):
                soybean_ratio = train_subset['soybean_close_lag1'].notna().sum() / len(train_subset)
                if soybean_ratio > 0.5:  # 학습 데이터의 50% 이상에 Soybean이 있어야 사용
                    use_soybean = True
                    regressors.append('soybean_close_lag1')
            
            # NaN 제거 (사용하는 regressors 기준)
            train_subset = train_subset.dropna(subset=regressors)
            
            # Prophet 모델 생성
            model = Prophet(
                seasonality_mode=prophet_config["seasonality_mode"],
                changepoint_prior_scale=prophet_config["changepoint_prior_scale"],
                yearly_seasonality=prophet_config["yearly_seasonality"],
                weekly_seasonality=prophet_config["weekly_seasonality"],
                daily_seasonality=prophet_config["daily_seasonality"],
            )
            
            # Regressors 추가
            for col in regressors:
                model.add_regressor(col, mode=prophet_config["regressor_mode"])
            
            # 모델 학습 - 데이터 타입 명시적 변환
            train_data = train_subset[["ds", "y"] + regressors].copy()
            train_data["ds"] = pd.to_datetime(train_data["ds"])
            train_data["y"] = pd.to_numeric(train_data["y"], errors='coerce').astype(float)
            for reg in regressors:
                train_data[reg] = pd.to_numeric(train_data[reg], errors='coerce').astype(float)
            
            model.fit(train_data)
            
            # 예측 - 데이터 타입 명시적 변환 (BigQuery 호환성)
            future = test_subset[["ds"] + regressors].copy()
            
            # ds는 datetime으로, regressors는 float로 확실하게 변환
            future["ds"] = pd.to_datetime(future["ds"])
            for reg in regressors:
                future[reg] = pd.to_numeric(future[reg], errors='coerce').astype(float)
            
            forecast = model.predict(future)
            
            # Features 추출
            prophet_feat = forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]].copy()
            
            # Seasonality 추가
            if "weekly" in forecast.columns:
                prophet_feat["weekly"] = forecast["weekly"]
            if "yearly" in forecast.columns:
                prophet_feat["yearly"] = forecast["yearly"]
            
            # Extra regressors 추가
            if "extra_regressors_multiplicative" in forecast.columns:
                prophet_feat["extra_regressors_multiplicative"] = forecast["extra_regressors_multiplicative"]
            
            # Regressor effects 추가
            for reg in regressors:
                if reg in forecast.columns:
                    prophet_feat[f"{reg}_effect"] = forecast[reg]
            
            # y는 전날(i) 종가 사용 (lag 1)
            prophet_feat["y"] = df.iloc[i]["y"]
            
            # 실제 예측 대상 날짜의 종가도 저장 (타겟 생성용)
            prophet_feat["y_next"] = test_subset["y"].values[0]
            
            # 사용한 외부 변수 플래그 추가
            prophet_feat["used_corn"] = use_corn
            prophet_feat["used_soybean"] = use_soybean
            
            # 원본 데이터의 다른 컬럼들 추가
            for col in test_subset.columns:
                if col not in prophet_feat.columns and col not in ["ds", "y"]:
                    prophet_feat[col] = test_subset[col].values[0]
            
            prophet_features_list.append(prophet_feat)
            pbar.update(1)
    
    # DataFrame으로 변환
    prophet_features_df = pd.concat(prophet_features_list, ignore_index=True)
    
    print(f"\n✅ Prophet features 추출 완료: {len(prophet_features_df)} 행")
    if has_corn:
        corn_used = prophet_features_df['used_corn'].sum()
        print(f"   - Corn 포함 예측: {corn_used}일")
    if has_soybean:
        soybean_used = prophet_features_df['used_soybean'].sum()
        print(f"   - Soybean 포함 예측: {soybean_used}일")
    
    return prophet_features_df


def create_target_variable(df):
    """
    타겟 변수 생성 (전날 대비 상승=1, 하락=0)
    y: 전날 종가
    y_next: 다음날 종가 (예측 대상)
    """
    df = df.copy()
    
    # y_next가 있는 경우 (y는 전날, y_next는 다음날)
    if 'y_next' in df.columns:
        # 전날(y) 대비 다음날(y_next) 변화량
        df['y_change'] = df['y_next'] - df['y']
        
        # 방향 (상승=1, 하락=0)
        df['direction'] = (df['y_change'] > 0).astype(int)
        
        print(f"✅ 타겟 변수 생성 (y=전날, y_next=예측날)")
    else:
        # 기존 방식 (하위 호환성)
        df['y_change'] = df['y'].diff()
        df['direction'] = (df['y_change'] > 0).astype(int)
        print(f"⚠️  y_next가 없어 기존 방식으로 타겟 생성")
    
    # 보합(변화 없음) 제거
    df = df[df['y_change'] != 0].copy()
    
    # NaN 제거
    df = df.dropna(subset=['y_change']).reset_index(drop=True)
    
    # Volatility 추가 (yhat_upper - yhat_lower)
    if 'yhat_upper' in df.columns and 'yhat_lower' in df.columns:
        df['volatility'] = df['yhat_upper'] - df['yhat_lower']
        print(f"✅ Volatility 컬럼 추가 완료 (yhat_upper - yhat_lower)")
    
    print(f"✅ 타겟 변수 생성 완료: {len(df)} 행 (보합 제외)")
    print(f"   - 상승(1): {(df['direction'] == 1).sum()}개")
    print(f"   - 하락(0): {(df['direction'] == 0).sum()}개\n")
    
    return df


def main(project_id=None, dataset_id=None, wheat_table_id=None, corn_table_id=None, soybean_table_id=None, end_date="2026-02-03", output_csv=None):
    """
    메인 실행 함수
    
    Args:
        project_id: GCP 프로젝트 ID (None이면 .env에서 VERTEX_AI_PROJECT_ID 또는 GCP_PROJECT_ID 사용)
        dataset_id: BigQuery 데이터셋 ID (None이면 .env에서 BIGQUERY_DATASET_ID 사용)
        wheat_table_id: Wheat 가격 테이블 ID (None이면 기본값 'wheat_price' 사용)
        corn_table_id: Corn 가격 테이블 ID (None이면 Corn 없이 진행)
        soybean_table_id: Soybean 가격 테이블 ID (None이면 Soybean 없이 진행)
        end_date: 데이터 종료 날짜 (기본값: 2026-02-03)
        output_csv: 출력 CSV 파일 경로 (기본값: prophet_features_wheat_YYYYMMDD_granger.csv)
    
    Note:
        - Granger causality 검증 결과를 반영한 Prophet feature 추출
        - Wheat Close ← Corn Close (lag 2)
        - Wheat Close ← Soybean Close (lag 1)
        - Wheat Close ← Wheat EMA (lag 1)
        - Wheat Close ← Wheat Volume (lag 1)
    """
    print("\n" + "=" * 80)
    print("🌾 Wheat Price Prediction with Granger Causality Features (BigQuery)")
    print("=" * 80)
    print("\n📋 Granger 검증 결과:")
    print("  - Wheat Close ← Corn Close (lag 2)")
    print("  - Wheat Close ← Soybean Close (lag 1)")
    print("  - Wheat Close ← Wheat EMA (lag 1)")
    print("  - Wheat Close ← Wheat Volume (lag 1)")
    print("=" * 80 + "\n")
    
    # 환경변수에서 설정 읽기
    if project_id is None:
        project_id = os.getenv("VERTEX_AI_PROJECT_ID") or os.getenv("GCP_PROJECT_ID")
        if not project_id:
            raise ValueError(
                "프로젝트 ID가 지정되지 않았습니다. "
                "인자로 전달하거나 .env 파일에 VERTEX_AI_PROJECT_ID 또는 GCP_PROJECT_ID를 설정하세요."
            )
    
    if dataset_id is None:
        dataset_id = os.getenv("BIGQUERY_DATASET_ID")
        if not dataset_id:
            raise ValueError(
                "데이터셋 ID가 지정되지 않았습니다. "
                "인자로 전달하거나 .env 파일에 BIGQUERY_DATASET_ID를 설정하세요."
            )
    
    if wheat_table_id is None:
        wheat_table_id = "wheat_price"
    
    print(f"📋 설정 정보:")
    print(f"   Project ID: {project_id}")
    print(f"   Dataset ID: {dataset_id}")
    print(f"   Wheat Table ID: {wheat_table_id}")
    if corn_table_id:
        print(f"   Corn Table ID: {corn_table_id}")
    if soybean_table_id:
        print(f"   Soybean Table ID: {soybean_table_id}")
    print(f"   End Date: {end_date}\n")
    
    # 설정 로드
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")
    config = load_config(config_path)
    
    # BigQuery에서 데이터 로드 (Wheat + Corn + Soybean)
    df = load_data_from_bigquery(project_id, dataset_id, wheat_table_id, corn_table_id, soybean_table_id, end_date)
    
    # 데이터 전처리
    df = preprocess_data(df)
    
    # Granger 검증 기반 Lag features 생성
    df = create_granger_lag_features(df)
    
    # Prophet features 추출
    prophet_features_df = extract_prophet_features_walkforward(df, config)
    
    # 타겟 변수 생성
    prophet_features_df = create_target_variable(prophet_features_df)
    
    # 성능 평가
    print("\n" + "=" * 80)
    print("📊 모델 성능 평가")
    print("=" * 80)
    
    # 1. Accuracy (방향 예측 정확도)
    if 'yhat' in prophet_features_df.columns and 'y' in prophet_features_df.columns and 'y_next' in prophet_features_df.columns:
        # yhat으로 예측한 방향 vs 실제 방향
        prophet_features_df['predicted_direction'] = (prophet_features_df['yhat'] > prophet_features_df['y']).astype(int)
        prophet_features_df['actual_direction'] = prophet_features_df['direction']
        
        accuracy = (prophet_features_df['predicted_direction'] == prophet_features_df['actual_direction']).mean()
        correct_predictions = (prophet_features_df['predicted_direction'] == prophet_features_df['actual_direction']).sum()
        total_predictions = len(prophet_features_df)
        
        print(f"🎯 Accuracy (방향 예측 정확도)")
        print(f"   - 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   - 맞춘 예측: {correct_predictions}/{total_predictions}")
        
        # 상승/하락 각각의 정확도
        up_mask = prophet_features_df['actual_direction'] == 1
        down_mask = prophet_features_df['actual_direction'] == 0
        
        if up_mask.sum() > 0:
            up_accuracy = (prophet_features_df[up_mask]['predicted_direction'] == 1).mean()
            print(f"   - 상승 예측 정확도: {up_accuracy:.4f} ({up_accuracy*100:.2f}%)")
        
        if down_mask.sum() > 0:
            down_accuracy = (prophet_features_df[down_mask]['predicted_direction'] == 0).mean()
            print(f"   - 하락 예측 정확도: {down_accuracy:.4f} ({down_accuracy*100:.2f}%)")
    
    # 2. MAE (Mean Absolute Error)
    if 'yhat' in prophet_features_df.columns and 'y_next' in prophet_features_df.columns:
        mae = np.abs(prophet_features_df['yhat'] - prophet_features_df['y_next']).mean()
        
        print(f"\n📏 MAE (Mean Absolute Error)")
        print(f"   - MAE: {mae:.4f}")
        print(f"   - 평균 예측 오차: ${mae:.2f}")
        
        # 추가 통계
        mean_actual = prophet_features_df['y_next'].mean()
        mape = (np.abs(prophet_features_df['yhat'] - prophet_features_df['y_next']) / prophet_features_df['y_next']).mean() * 100
        print(f"   - MAPE: {mape:.2f}%")
        print(f"   - 실제 가격 평균: ${mean_actual:.2f}")
    
    print("=" * 80 + "\n")
    
    # 출력 파일명 결정 (granger 표시 추가)
    if output_csv is None:
        output_csv = f"prophet_features_wheat_{end_date.replace('-', '')}_granger.csv"
    
    # CSV 저장
    output_path = os.path.join(base_dir, output_csv)
    prophet_features_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 80)
    print("📊 추출된 Features 요약 (Granger 검증 기반 - Wheat)")
    print("=" * 80)
    print(f"총 행 수: {len(prophet_features_df)}")
    print(f"총 컬럼 수: {len(prophet_features_df.columns)}")
    
    print(f"\n🔄 데이터 구조:")
    print(f"  - ds: 예측 대상 날짜")
    print(f"  - y: 전날 종가 (lag 1)")
    print(f"  - y_next: 예측 대상 날짜의 실제 종가")
    print(f"  - direction: y → y_next 방향 (1=상승, 0=하락)")
    print(f"  - volatility: 변동성 (yhat_upper - yhat_lower)")
    
    print(f"\n🔬 Granger Features:")
    granger_cols = [col for col in prophet_features_df.columns if 'lag' in col or 'corn' in col or 'soybean' in col]
    for col in granger_cols:
        if col in prophet_features_df.columns:
            print(f"  - {col}")
    
    print(f"\n전체 Features: {list(prophet_features_df.columns)}")
    print(f"\n타겟 분포:")
    print(f"  상승(1): {(prophet_features_df['direction'] == 1).sum()}개 ({(prophet_features_df['direction'] == 1).mean()*100:.1f}%)")
    print(f"  하락(0): {(prophet_features_df['direction'] == 0).sum()}개 ({(prophet_features_df['direction'] == 0).mean()*100:.1f}%)")
    
    # 성능 요약
    if 'predicted_direction' in prophet_features_df.columns:
        accuracy = (prophet_features_df['predicted_direction'] == prophet_features_df['actual_direction']).mean()
        print(f"\n성능 요약:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if 'yhat' in prophet_features_df.columns and 'y_next' in prophet_features_df.columns:
        mae = np.abs(prophet_features_df['yhat'] - prophet_features_df['y_next']).mean()
        print(f"  MAE: {mae:.4f}")
    
    print(f"\n✅ 결과 저장 완료: {output_path}")
    print("=" * 80 + "\n")
    
    return prophet_features_df


if __name__ == "__main__":
    # .env 파일에서 자동으로 설정을 읽어옵니다
    # 필요시 인자로 직접 전달할 수도 있습니다
    
    # 기본값: .env 파일의 설정 사용
    # Wheat + Corn + Soybean 데이터 사용 (Granger causality 반영)
    results = main(
        wheat_table_id="wheat_price",
        corn_table_id="corn_price",
        soybean_table_id="soybean_price",
        end_date="2026-02-03"
    )
    
    # Corn, Soybean 없이 실행 (EMA, Volume만 사용):
    # results = main(
    #     wheat_table_id="wheat_price",
    #     corn_table_id=None,
    #     soybean_table_id=None,
    #     end_date="2026-02-03"
    # )
