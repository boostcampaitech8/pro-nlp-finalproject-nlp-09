import pandas as pd
import numpy as np
from prophet import Prophet
import yaml
from tqdm import tqdm
import warnings
import random
from google.cloud import bigquery
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
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_data_from_bigquery(project_id, dataset_id, corn_table_id, soybean_table_id=None, end_date="2026-02-03"):
    print(f"📊 BigQuery에서 데이터 로딩 중...")
    client = bigquery.Client(project=project_id)

    query_corn = f"""
    SELECT 
        time,
        open,
        high,
        low,
        close,
        ema as EMA,
        volume as Volume
    FROM `{project_id}.{dataset_id}.{corn_table_id}`
    WHERE DATE(time) <= '{end_date}'
    ORDER BY time
    """
    
    try:
        df_corn = client.query(query_corn).to_dataframe()
    except Exception as e:
        # 컬럼명이 대문자일 경우를 대비한 재시도
        print(f"   첫 번째 쿼리 실패, 대체 쿼리 시도...")
        query_corn = f"""
        SELECT 
            time,
            open,
            high,
            low,
            close,
            EMA,
            Volume
        FROM `{project_id}.{dataset_id}.{corn_table_id}`
        WHERE DATE(time) <= '{end_date}'
        ORDER BY time
        """
        df_corn = client.query(query_corn).to_dataframe()
    
    print(f"✅ Corn 데이터 로드 완료: {len(df_corn)} 행")

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

            df = pd.merge(df_corn, df_soybean, on='time', how='left')
            
            missing_count = df['soybean_close'].isna().sum()
            print(f"\n✅ 데이터 병합 완료: {len(df)} 행 (Corn 기준)")
        except Exception as e:
            print(f"⚠️  Soybean 데이터 로드 실패: {e}")
            print(f"   Soybean 없이 진행합니다.\n")
            df = df_corn
    else:
        df = df_corn
        print("\n⚠️  Soybean 데이터 없이 진행합니다.\n")
    
    return df


def preprocess_data(df):
    df["ds"] = pd.to_datetime(df["time"])
    df["y"] = pd.to_numeric(df["close"], errors='coerce')

    df["Volume"] = pd.to_numeric(df["Volume"], errors='coerce')
    df["EMA"] = pd.to_numeric(df["EMA"], errors='coerce')

    if 'soybean_close' in df.columns:
        df["soybean_close"] = pd.to_numeric(df["soybean_close"], errors='coerce')
        df = df[["ds", "y", "Volume", "EMA", "soybean_close"]].copy()
    else:
        df = df[["ds", "y", "Volume", "EMA"]].copy()
    
    df = df.sort_values("ds").reset_index(drop=True)

    if 'soybean_close' in df.columns:
        df = df.dropna(subset=['ds', 'y', 'Volume', 'EMA']).reset_index(drop=True)
        soybean_available = df['soybean_close'].notna().sum()
        soybean_missing = df['soybean_close'].isna().sum()
        print(f"✅ 데이터 전처리 완료: {len(df)} 행")
    else:
        df = df.dropna().reset_index(drop=True)
        print(f"✅ 데이터 전처리 완료: {len(df)} 행 (Corn 데이터만)\n")
    
    return df


def create_granger_lag_features(df):
    df = df.copy()
    
    if 'soybean_close' in df.columns:
        df['soybean_close_lag8'] = df['soybean_close'].shift(8).astype(float)
    
    df['EMA_lag2'] = df['EMA'].shift(2).astype(float)
    df['Volume_lag5'] = df['Volume'].shift(5).astype(float)

    original_len = len(df)
    df = df.dropna(subset=['EMA_lag2', 'Volume_lag5']).reset_index(drop=True)
    
    if 'soybean_close_lag8' in df.columns:
        soybean_available = df['soybean_close_lag8'].notna().sum()
        soybean_missing = df['soybean_close_lag8'].isna().sum()
        print(f"\n✅ Lag Features 생성 완료: {len(df)} 행")
    else:
        print(f"\n✅ Lag Features 생성 완료: {len(df)} 행 (Soybean 없이 진행)\n")
    
    return df


def extract_prophet_features_walkforward(df, config, start_date=None, end_date=None):
    prophet_config = config["prophet"]
    train_window_days = int(prophet_config["train_window_years"] * 365)
    
    # 예측 대상 날짜 필터링
    predict_start_idx = 0
    predict_end_idx = len(df) - 1
    
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        predict_start_idx = df[df['ds'] >= start_date].index.min()
        if pd.isna(predict_start_idx):
            print(f"⚠️  경고: start_date '{start_date}'에 해당하는 데이터가 없습니다.")
            predict_start_idx = 0
        else:
            print(f"📅 예측 시작 날짜: {df.loc[predict_start_idx, 'ds'].strftime('%Y-%m-%d')}")
    
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        predict_end_idx = df[df['ds'] <= end_date].index.max()
        if pd.isna(predict_end_idx):
            print(f"⚠️  경고: end_date '{end_date}'에 해당하는 데이터가 없습니다.")
            predict_end_idx = len(df) - 1
        else:
            print(f"📅 예측 종료 날짜: {df.loc[predict_end_idx, 'ds'].strftime('%Y-%m-%d')}")

    base_regressors = ['EMA_lag2', 'Volume_lag5']
    has_soybean = 'soybean_close_lag8' in df.columns
    
    prophet_features_list = []

    effective_start_idx = max(train_window_days, predict_start_idx)
    effective_end_idx = min(predict_end_idx, len(df) - 2)

    with tqdm(total=effective_end_idx - effective_start_idx + 1, desc="Prophet 학습 및 예측") as pbar:
        for i in range(effective_start_idx, effective_end_idx + 1):
            train_subset = df.iloc[max(0, i - train_window_days) : i].copy()
            test_subset = df.iloc[i + 1 : i + 2].copy()

            use_soybean = False
            if has_soybean and pd.notna(test_subset['soybean_close_lag8'].values[0]):
                soybean_ratio = train_subset['soybean_close_lag8'].notna().sum() / len(train_subset)
                if soybean_ratio > 0.5:
                    use_soybean = True
                    train_subset = train_subset.dropna(subset=['soybean_close_lag8'])

            if use_soybean:
                regressors = base_regressors + ['soybean_close_lag8']
            else:
                regressors = base_regressors
            
            model = Prophet(
                seasonality_mode=prophet_config["seasonality_mode"],
                changepoint_prior_scale=prophet_config["changepoint_prior_scale"],
                yearly_seasonality=prophet_config["yearly_seasonality"],
                weekly_seasonality=prophet_config["weekly_seasonality"],
                daily_seasonality=prophet_config["daily_seasonality"],
            )
            
            for col in regressors:
                model.add_regressor(col, mode=prophet_config["regressor_mode"])
            
            train_data = train_subset[["ds", "y"] + regressors].copy()
            train_data["ds"] = pd.to_datetime(train_data["ds"])
            train_data["y"] = pd.to_numeric(train_data["y"], errors='coerce').astype(float)
            for reg in regressors:
                train_data[reg] = pd.to_numeric(train_data[reg], errors='coerce').astype(float)
            
            model.fit(train_data)
            
            future = test_subset[["ds"] + regressors].copy()

            future["ds"] = pd.to_datetime(future["ds"])
            for reg in regressors:
                future[reg] = pd.to_numeric(future[reg], errors='coerce').astype(float)
            
            forecast = model.predict(future)
            
            prophet_feat = forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]].copy()

            if "weekly" in forecast.columns:
                prophet_feat["weekly"] = forecast["weekly"]
            if "yearly" in forecast.columns:
                prophet_feat["yearly"] = forecast["yearly"]
            
            if "extra_regressors_multiplicative" in forecast.columns:
                prophet_feat["extra_regressors_multiplicative"] = forecast["extra_regressors_multiplicative"]
            
            for reg in regressors:
                if reg in forecast.columns:
                    prophet_feat[f"{reg}_effect"] = forecast[reg]
            
            prophet_feat["y"] = df.iloc[i]["y"]
           
            prophet_feat["used_soybean"] = use_soybean
            
            for col in test_subset.columns:
                if col not in prophet_feat.columns and col not in ["ds", "y"]:
                    prophet_feat[col] = test_subset[col].values[0]
            
            prophet_features_list.append(prophet_feat)
            pbar.update(1)
    
    prophet_features_df = pd.concat(prophet_features_list, ignore_index=True)
    
    print(f"\n✅ Prophet features 추출 완료: {len(prophet_features_df)} 행")
    
    return prophet_features_df


def create_target_variable(df):
    df = df.copy()
    
    if 'y_next' in df.columns:
        df['y_change'] = df['y_next'] - df['y']
        df['direction'] = (df['y_change'] > 0).astype(int)
        
    df['y_change'] = df['y'].diff()
    df['direction'] = (df['y_change'] > 0).astype(int)
    df = df[df['y_change'] != 0].copy()
    df = df.dropna(subset=['y_change']).reset_index(drop=True)
    
    df['volatility'] = df['yhat_upper'] - df['yhat_lower']
    return df


def main(project_id=None, dataset_id=None, corn_table_id=None, soybean_table_id=None, end_date="2026-02-03", output_csv=None, start_date=None):
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
    
    if corn_table_id is None:
        corn_table_id = os.getenv("BIGQUERY_TABLE_ID") or "corn_price"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")
    config = load_config(config_path)
    
    df = load_data_from_bigquery(project_id, dataset_id, corn_table_id, soybean_table_id, end_date)
    
    df = preprocess_data(df)
    
    df = create_granger_lag_features(df)
    
    prophet_features_df = extract_prophet_features_walkforward(df, config, start_date=start_date)
    
    prophet_features_df = create_target_variable(prophet_features_df)
    
    if output_csv is None:
        output_csv = f"prophet_features_{end_date.replace('-', '')}_granger.csv"
    
    output_path = os.path.join(base_dir, output_csv)
    prophet_features_df.to_csv(output_path, index=False)
    
    return prophet_features_df


if __name__ == "__main__":
    start_date = '2010-09-22'
    end_date = '2026-02-06'
    
    results = main(
        corn_table_id="corn_price",
        soybean_table_id="soybean_price",  
        end_date=end_date,
        start_date=start_date
    )
