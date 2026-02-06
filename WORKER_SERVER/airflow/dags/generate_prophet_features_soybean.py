from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from prophet import Prophet
import logging
from google.cloud import bigquery
import os

# [환경 설정]
PROJECT_ID = os.getenv("VERTEX_AI_PROJECT_ID", "team-blue-448407")
DATASET_ID = "tilda"
PRICE_TABLE = "soybean_price"
TARGET_TABLE = "prophet_soybean"
COMMODITY = "soybean"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def get_bq_client():
    return bigquery.Client(project=PROJECT_ID)

def check_new_data(**context):
    client = get_bq_client()
    try:
        query_target = f"SELECT MAX(ds) as max_date FROM `{PROJECT_ID}.{DATASET_ID}.{TARGET_TABLE}`"
        df_target = client.query(query_target).to_dataframe()
        last_loaded_date = df_target['max_date'].iloc[0]
    except Exception:
        last_loaded_date = None
        
    query_source = f"SELECT MAX(DATE(time)) as max_date FROM `{PROJECT_ID}.{DATASET_ID}.{PRICE_TABLE}`"
    df_source = client.query(query_source).to_dataframe()
    latest_source_date = df_source['max_date'].iloc[0]
    
    if not latest_source_date: return False

    if last_loaded_date is None or latest_source_date > last_loaded_date:
        context['ti'].xcom_push(key='target_date', value=str(latest_source_date))
        return True
    return False

def generate_features(**context):
    is_update_needed = context['ti'].xcom_pull(task_ids='check_new_data')
    if not is_update_needed: return None

    target_date_str = context['ti'].xcom_pull(key='target_date', task_ids='check_new_data')
    target_date = pd.to_datetime(target_date_str)
    client = get_bq_client()
    
    start_date = target_date - timedelta(days=365 * 4 + 30)
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    # 1. Soybean 데이터 로드
    q_soy = f"SELECT time, close, volume as Volume, ema as EMA FROM `{PROJECT_ID}.{DATASET_ID}.{PRICE_TABLE}` WHERE DATE(time) >= '{start_date_str}' AND DATE(time) <= '{target_date_str}' ORDER BY time"
    df = client.query(q_soy).to_dataframe()
    df['ds'] = pd.to_datetime(df['time'])
    df['y'] = pd.to_numeric(df['close'])
    
    # 2. Corn & Wheat 데이터 로드 (Granger 검증 기반)
    # Corn: lag 6, Wheat: lag 1
    for dep in ['corn', 'wheat']:
        q_dep = f"SELECT time, close as {dep}_close FROM `{PROJECT_ID}.{DATASET_ID}.{dep}_price` WHERE DATE(time) >= '{start_date_str}' AND DATE(time) <= '{target_date_str}' ORDER BY time"
        try:
            df_dep = client.query(q_dep).to_dataframe()
            df_dep['time'] = pd.to_datetime(df_dep['time'])
            df = pd.merge(df, df_dep, on='time', how='left')
        except Exception as e:
            logging.warning(f"{dep} 데이터 로드 실패: {e}")
            df[f'{dep}_close'] = np.nan

    # 3. 피처 엔지니어링 (Lag 생성)
    # Soybean: Volume_lag1, EMA_lag1, corn_close_lag6, wheat_close_lag1
    df['Volume_lag1'] = df['Volume'].shift(1)
    df['EMA_lag1'] = df['EMA'].shift(1)
    df['corn_close_lag6'] = df['corn_close'].shift(6)
    df['wheat_close_lag1'] = df['wheat_close'].shift(1)
    
    train_df = df[df['ds'] < target_date].dropna(subset=['Volume_lag1', 'EMA_lag1']).copy()
    
    # 조건부 Regressors 체크 (데이터 충분성)
    base_regressors = ['Volume_lag1', 'EMA_lag1']
    extra_regressors = []
    if 'corn_close_lag6' in train_df.columns and train_df['corn_close_lag6'].notna().mean() > 0.5:
        extra_regressors.append('corn_close_lag6')
    if 'wheat_close_lag1' in train_df.columns and train_df['wheat_close_lag1'].notna().mean() > 0.5:
        extra_regressors.append('wheat_close_lag1')
        
    regressors = base_regressors + extra_regressors
    logging.info(f"Regressors: {regressors}")
    
    # 4. Prophet 모델 학습
    model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    for reg in regressors: model.add_regressor(reg, mode='multiplicative')
    model.fit(train_df[['ds', 'y'] + regressors])
    
    # 5. 예측
    future_row = df[df['ds'] == target_date].copy()
    if future_row[regressors].isna().any().any():
        future_row[regressors] = future_row[regressors].fillna(method='ffill')
        
    forecast = model.predict(future_row)
    
    # 6. 결과 정리
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
    
    # Lag 값들
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
    row = {k: (None if pd.isna(v) else v) for k, v in feature_data.items()}
    errors = client.insert_rows_json(table_id, [row])
    if errors: raise RuntimeError(f"BigQuery 적재 실패: {errors}")

with DAG(
    'generate_prophet_features_soybean_v1',
    default_args=default_args,
    description='Soybean 가격 데이터를 가공하여 Prophet 피처 생성 및 적재',
    schedule_interval='40 16 * * *', # 매일 16:40
    catchup=False,
    tags=['soybean', 'prophet', 'feature_engineering']
) as dag:

    t1 = PythonOperator(task_id='check_new_data', python_callable=check_new_data, provide_context=True)
    t2 = PythonOperator(task_id='generate_features', python_callable=generate_features, provide_context=True)
    t3 = PythonOperator(task_id='insert_to_bq', python_callable=insert_to_bq, provide_context=True)

    t1 >> t2 >> t3
