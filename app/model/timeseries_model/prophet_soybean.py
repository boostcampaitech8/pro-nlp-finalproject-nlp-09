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


def load_data_from_bigquery(project_id, dataset_id, soybean_table_id, corn_table_id=None, wheat_table_id=None, end_date="2026-02-03"):
    print(f"📊 BigQuery에서 데이터 로딩 중...")
    client = bigquery.Client(project=project_id)
    query_soybean = f"""
    SELECT 
        time,
        open,
        high,
        low,
        close,
        ema as EMA,
        volume as Volume
    FROM `{project_id}.{dataset_id}.{soybean_table_id}`
    WHERE DATE(time) <= '{end_date}'
    ORDER BY time
    """
    
    try:
        df_soybean = client.query(query_soybean).to_dataframe()
    except Exception as e:
        print(f"   첫 번째 쿼리 실패, 대체 쿼리 시도...")
        query_soybean = f"""
        SELECT 
            time,
            open,
            high,
            low,
            close,
            EMA,
            Volume
        FROM `{project_id}.{dataset_id}.{soybean_table_id}`
        WHERE DATE(time) <= '{end_date}'
        ORDER BY time
        """
        df_soybean = client.query(query_soybean).to_dataframe()
    
    print(f"✅ Soybean 데이터 로드 완료: {len(df_soybean)} 행")
    print(f"   기간: {df_soybean['time'].min()} ~ {df_soybean['time'].max()}")
    
    df = df_soybean.copy()

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

            df = pd.merge(df, df_corn, on='time', how='left')
            
        except Exception as e:
            print(f"⚠️  Corn 데이터 로드 실패: {e}")
            print(f"   Corn 없이 진행합니다.")
    
    if wheat_table_id:
        print(f"\n📊 Wheat 데이터 로딩 중...")
        query_wheat = f"""
        SELECT 
            time,
            close as wheat_close
        FROM `{project_id}.{dataset_id}.{wheat_table_id}`
        WHERE DATE(time) <= '{end_date}'
        ORDER BY time
        """
        
        try:
            df_wheat = client.query(query_wheat).to_dataframe()
            print(f"✅ Wheat 데이터 로드 완료: {len(df_wheat)} 행")
            print(f"   기간: {df_wheat['time'].min()} ~ {df_wheat['time'].max()}")

            df = pd.merge(df, df_wheat, on='time', how='left')
            
        except Exception as e:
            print(f"⚠️  Wheat 데이터 로드 실패: {e}")
            print(f"   Wheat 없이 진행합니다.")
    
    return df


def preprocess_data(df):
    df["ds"] = pd.to_datetime(df["time"])
    df["y"] = pd.to_numeric(df["close"], errors='coerce')
    
    df["Volume"] = pd.to_numeric(df["Volume"], errors='coerce')
    df["EMA"] = pd.to_numeric(df["EMA"], errors='coerce')
    
    cols = ["ds", "y", "Volume", "EMA"]
    
    if 'corn_close' in df.columns:
        df["corn_close"] = pd.to_numeric(df["corn_close"], errors='coerce')
        cols.append("corn_close")
    
    if 'wheat_close' in df.columns:
        df["wheat_close"] = pd.to_numeric(df["wheat_close"], errors='coerce')
        cols.append("wheat_close")
    
    df = df[cols].copy()
    df = df.sort_values("ds").reset_index(drop=True)
    
    df = df.dropna(subset=['ds', 'y', 'Volume', 'EMA']).reset_index(drop=True)
    
    return df


def create_granger_lag_features(df):
    df = df.copy()
    
    if 'corn_close' in df.columns:
        df['corn_close_lag6'] = df['corn_close'].shift(6).astype(float)
    
    if 'wheat_close' in df.columns:
        df['wheat_close_lag1'] = df['wheat_close'].shift(1).astype(float)
    
    df['Volume_lag1'] = df['Volume'].shift(1).astype(float)
    df['EMA_lag1'] = df['EMA'].shift(1).astype(float)

    df = df.dropna(subset=['Volume_lag1', 'EMA_lag1']).reset_index(drop=True)
    
    return df


def extract_prophet_features_walkforward(df, config, start_date=None, end_date=None):
    prophet_config = config["prophet"]
    train_window_days = int(prophet_config["train_window_years"] * 365)
    
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

    base_regressors = ['Volume_lag1', 'EMA_lag1']
    has_corn = 'corn_close_lag6' in df.columns
    has_wheat = 'wheat_close_lag1' in df.columns
    
    prophet_features_list = []

    effective_start_idx = max(train_window_days, predict_start_idx)
    effective_end_idx = min(predict_end_idx, len(df) - 2)
    
    with tqdm(total=effective_end_idx - effective_start_idx + 1, desc="Prophet 학습 및 예측") as pbar:
        for i in range(effective_start_idx, effective_end_idx + 1):
            train_start_idx = max(0, i - train_window_days)
            train_end_idx = i
            train_subset = df.iloc[train_start_idx:train_end_idx].copy()
            test_subset = df.iloc[i + 1 : i + 2].copy()
            
            regressors = base_regressors.copy()
            
            use_corn = False
            if has_corn and pd.notna(test_subset['corn_close_lag6'].values[0]):
                corn_ratio = train_subset['corn_close_lag6'].notna().sum() / len(train_subset)
                if corn_ratio > 0.5:
                    use_corn = True
                    regressors.append('corn_close_lag6')
            
            use_wheat = False
            if has_wheat and pd.notna(test_subset['wheat_close_lag1'].values[0]):
                wheat_ratio = train_subset['wheat_close_lag1'].notna().sum() / len(train_subset)
                if wheat_ratio > 0.5:
                    use_wheat = True
                    regressors.append('wheat_close_lag1')
            
            train_subset = train_subset.dropna(subset=regressors)
            
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
            
            prophet_feat["y_next"] = test_subset["y"].values[0]
            
            prophet_feat["used_corn"] = use_corn
            prophet_feat["used_wheat"] = use_wheat
            
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
        
    else:
        df['y_change'] = df['y'].diff()
        df['direction'] = (df['y_change'] > 0).astype(int)
        print(f"⚠️  y_next가 없어 기존 방식으로 타겟 생성")
    
    df = df[df['y_change'] != 0].copy()
    
    df = df.dropna(subset=['y_change']).reset_index(drop=True)
    
    if 'yhat_upper' in df.columns and 'yhat_lower' in df.columns:
        df['volatility'] = df['yhat_upper'] - df['yhat_lower']
    
    return df


def main(project_id=None, dataset_id=None, soybean_table_id=None, corn_table_id=None, wheat_table_id=None, end_date="2026-02-03", output_csv=None, start_date=None):
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
    
    if soybean_table_id is None:
        soybean_table_id = "soybean_price"
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")
    config = load_config(config_path)
    
    df = load_data_from_bigquery(project_id, dataset_id, soybean_table_id, corn_table_id, wheat_table_id, end_date)
    
    df = preprocess_data(df)
    
    df = create_granger_lag_features(df)
    
    prophet_features_df = extract_prophet_features_walkforward(df, config, start_date=start_date)
    
    prophet_features_df = create_target_variable(prophet_features_df)
    
    if output_csv is None:
        output_csv = f"prophet_features_soybean_{end_date.replace('-', '')}_granger.csv"
    
    output_path = os.path.join(base_dir, output_csv)
    prophet_features_df.to_csv(output_path, index=False)
    print(f"💾 저장 완료: {output_path}")
    
    return prophet_features_df


if __name__ == "__main__":
    start_date = '2015-04-22'
    end_date = '2026-02-06'
    
    results = main(
        soybean_table_id="soybean_price",
        corn_table_id="corn_price",
        wheat_table_id="wheat_price",
        end_date=end_date,
        start_date=start_date
    )
