from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from prophet import Prophet
import logging
from google.cloud import bigquery
import os

PROJECT_ID = os.getenv("VERTEX_AI_PROJECT_ID", "project-5b75bb04-485d-454e-af7")
DATASET_ID = "tilda"
PRICE_TABLE = "soybean_price"
TARGET_TABLE = "prophet_soybean"
COMMODITY = "soybean"
START_DATE = datetime(2024, 1, 1)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': START_DATE,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def get_bq_client():
    return bigquery.Client(project=PROJECT_ID)

def generate_features(**context):
    target_date_str = context['ds']
    target_date = pd.to_datetime(target_date_str)
    logging.info(f"Target Date: {target_date_str} 피처 생성 시작")
    
    client = get_bq_client()
    start_date = target_date - timedelta(days=365 * 4 + 30)
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    # 1. Soybean 데이터
    q_soy = f"SELECT time, close, volume as Volume, ema as EMA FROM `{PROJECT_ID}.{DATASET_ID}.{PRICE_TABLE}` WHERE DATE(time) >= '{start_date_str}' AND DATE(time) <= '{target_date_str}' ORDER BY time"
    df = client.query(q_soy).to_dataframe()
    
    if df.empty or df.iloc[-1]['time'].strftime('%Y-%m-%d') != target_date_str:
        logging.warning(f"{target_date_str} 데이터가 아직 없습니다.")
        return None

    df['ds'] = pd.to_datetime(df['time'])
    df['y'] = pd.to_numeric(df['close'])
    
    # 2. Corn & Wheat 데이터
    for dep in ['corn', 'wheat']:
        q_dep = f"SELECT time, close as {dep}_close FROM `{PROJECT_ID}.{DATASET_ID}.{dep}_price` WHERE DATE(time) >= '{start_date_str}' AND DATE(time) <= '{target_date_str}' ORDER BY time"
        try:
            df_dep = client.query(q_dep).to_dataframe()
            df_dep['time'] = pd.to_datetime(df_dep['time'])
            df = pd.merge(df, df_dep, on='time', how='left')
        except Exception as e:
            logging.warning(f"{dep} 데이터 로드 실패: {e}")
            df[f'{dep}_close'] = np.nan

    # 3. Lag 생성
    df['Volume_lag1'] = df['Volume'].shift(1)
    df['EMA_lag1'] = df['EMA'].shift(1)
    df['corn_close_lag6'] = df['corn_close'].shift(6)
    df['wheat_close_lag1'] = df['wheat_close'].shift(1)
    
    train_df = df[df['ds'] < target_date].dropna(subset=['Volume_lag1', 'EMA_lag1']).copy()
    
    base_regressors = ['Volume_lag1', 'EMA_lag1']
    extra_regressors = []
    if 'corn_close_lag6' in train_df.columns and train_df['corn_close_lag6'].notna().mean() > 0.5:
        extra_regressors.append('corn_close_lag6')
    if 'wheat_close_lag1' in train_df.columns and train_df['wheat_close_lag1'].notna().mean() > 0.5:
        extra_regressors.append('wheat_close_lag1')
        
    regressors = base_regressors + extra_regressors
    
    model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    for reg in regressors: model.add_regressor(reg, mode='multiplicative')
    model.fit(train_df[['ds', 'y'] + regressors])
    
    future_row = df[df['ds'] == target_date].copy()
    if future_row[regressors].isna().any().any():
        future_row[regressors] = future_row[regressors].fillna(method='ffill')
        
    forecast = model.predict(future_row)
    
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].iloc[0].to_dict()
    for col in ['weekly', 'yearly', 'extra_regressors_multiplicative']:
        if col in forecast.columns: result[col] = forecast[col].iloc[0]
    for reg in regressors:
        if reg in forecast.columns: result[f"{reg}_effect"] = forecast[reg].iloc[0]
            
    result['y'] = future_row['y'].iloc[0]
    result['y_next'] = None
    result['Volume'] = future_row['Volume'].iloc[0]
    result['EMA'] = future_row['EMA'].iloc[0]
    result['corn_close'] = future_row['corn_close'].iloc[0] if 'corn_close' in future_row else None
    result['wheat_close'] = future_row['wheat_close'].iloc[0] if 'wheat_close' in future_row else None
    result['Volume_lag1'] = future_row['Volume_lag1'].iloc[0]
    result['EMA_lag1'] = future_row['EMA_lag1'].iloc[0]
    result['corn_close_lag6'] = future_row['corn_close_lag6'].iloc[0] if 'corn_close_lag6' in future_row else None
    result['wheat_close_lag1'] = future_row['wheat_close_lag1'].iloc[0] if 'wheat_close_lag1' in future_row else None
    result['used_corn'] = 'corn_close_lag6' in extra_regressors
    result['used_wheat'] = 'wheat_close_lag1' in extra_regressors
    result['volatility'] = result['yhat_upper'] - result['yhat_lower']
    result['ds'] = result['ds'].strftime('%Y-%m-%d')
    
    return result

def insert_to_bq(**context):
    feature_data = context['ti'].xcom_pull(task_ids='generate_features')
    if not feature_data: return
    client = get_bq_client()
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TARGET_TABLE}"
    
    # 중복 방지 (삭제 후 삽입)
    try:
        client.query(f"DELETE FROM `{table_id}` WHERE ds = '{feature_data['ds']}'").result()
    except Exception: pass

    row = {k: (None if pd.isna(v) else v) for k, v in feature_data.items()}
    errors = client.insert_rows_json(table_id, [row])
    if errors: raise RuntimeError(f"BigQuery 적재 실패: {errors}")
    logging.info(f"적재 완료: {row['ds']}")

with DAG(
    'generate_prophet_features_soybean_v1',
    default_args=default_args,
    description='Soybean Prophet 피처 생성 (Backfill 지원)',
    schedule_interval='40 16 * * *',
    catchup=True,
    max_active_runs=1,
    tags=['soybean', 'prophet', 'feature_engineering']
) as dag:

    t1 = PythonOperator(task_id='generate_features', python_callable=generate_features, provide_context=True)
    t2 = PythonOperator(task_id='insert_to_bq', python_callable=insert_to_bq, provide_context=True)
    t1 >> t2