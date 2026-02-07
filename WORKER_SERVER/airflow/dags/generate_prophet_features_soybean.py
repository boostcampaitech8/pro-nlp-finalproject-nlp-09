from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from prophet import Prophet
import logging
from google.cloud import bigquery
import os
import json

# [환경 설정]
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

def check_already_exists(**context):
    """이미 해당 날짜의 데이터가 적재되어 있는지 확인"""
    target_date = context['ds']
    client = get_bq_client()
    query = f"SELECT count(*) as cnt FROM `{PROJECT_ID}.{DATASET_ID}.{TARGET_TABLE}` WHERE ds = '{target_date}'"
    try:
        df = client.query(query).to_dataframe()
        if df['cnt'].iloc[0] > 0:
            logging.info(f"[PRICE-MODEL][{COMMODITY}] ✅ {target_date} 데이터가 이미 존재합니다. 작업을 건너뜁니다.")
            return False
        return True
    except Exception as e:
        logging.warning(f"[PRICE-MODEL][{COMMODITY}] ⚠️ 체크 실패 (진행): {e}")
        return True

def generate_features(**context):
    """
    Airflow 실행 날짜(ds)를 기준으로 Prophet 모델을 학습하고 피처를 생성합니다.
    (prophet_soybean.py 로직과 100% 일치화)
    """
    target_date_str = context['ds']
    target_date = pd.to_datetime(target_date_str)
    
    logging.info(f"[PRICE-MODEL][{COMMODITY}] 🚀 Task 시작: generate_features (Target: {target_date_str})")
    
    client = get_bq_client()
    start_date = target_date - timedelta(days=365 * 4 + 30)
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    # 1. Soybean 데이터 로드
    q_soy = f"SELECT time, close, open, high, low, volume as Volume, ema as EMA FROM `{PROJECT_ID}.{DATASET_ID}.{PRICE_TABLE}` WHERE DATE(time) >= '{start_date_str}' AND DATE(time) <= '{target_date_str}' ORDER BY time"
    df = client.query(q_soy).to_dataframe()
    
    if df.empty or df.iloc[-1]['time'].strftime('%Y-%m-%d') != target_date_str:
        logging.warning(f"[PRICE-MODEL][{COMMODITY}] ⚠️ {target_date_str} 데이터 부재로 중단.")
        return None

    df['ds'] = pd.to_datetime(df['time'])
    df['y'] = pd.to_numeric(df['close'])
    
    # 2. 보조 데이터(Corn, Wheat) 로드 (Granger 기반)
    for dep in ['corn', 'wheat']:
        q_dep = f"SELECT time, close as {dep}_close FROM `{PROJECT_ID}.{DATASET_ID}.{dep}_price` WHERE DATE(time) >= '{start_date_str}' AND DATE(time) <= '{target_date_str}' ORDER BY time"
        try:
            df_dep = client.query(q_dep).to_dataframe()
            df_dep['time'] = pd.to_datetime(df_dep['time'])
            df = pd.merge(df, df_dep, on='time', how='left')
        except Exception: df[f'{dep}_close'] = np.nan

    # 3. Granger Lag 피처 생성
    df['Volume_lag1'] = df['Volume'].shift(1).astype(float)
    df['EMA_lag1'] = df['EMA'].shift(1).astype(float)
    df['corn_close_lag6'] = df['corn_close'].shift(6).astype(float)
    df['wheat_close_lag1'] = df['wheat_close'].shift(1).astype(float)
    
    # 학습 데이터 준비 (target_date 전날까지)
    train_df = df[df['ds'] < target_date].dropna(subset=['Volume_lag1', 'EMA_lag1']).copy()
    
    # 조건부 Regressors 체크
    use_corn, use_wheat = False, False
    regressors = ['Volume_lag1', 'EMA_lag1']
    
    if 'corn_close_lag6' in train_df.columns and train_df['corn_close_lag6'].notna().mean() > 0.5:
        use_corn = True
        regressors.append('corn_close_lag6')
    if 'wheat_close_lag1' in train_df.columns and train_df['wheat_close_lag1'].notna().mean() > 0.5:
        use_wheat = True
        regressors.append('wheat_close_lag1')
        
    train_df = train_df.dropna(subset=regressors)
    logging.info(f"[PRICE-MODEL][{COMMODITY}] 🧠 학습 데이터: {len(train_df)}행, Regressors: {regressors}")
    
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
    res = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].iloc[0].to_dict()
    for col in ['weekly', 'yearly', 'extra_regressors_multiplicative']:
        res[col] = forecast[col].iloc[0] if col in forecast.columns else None
    for reg in regressors:
        res[f"{reg}_effect"] = forecast[reg].iloc[0] if reg in forecast.columns else None
            
    prev_idx = df[df['ds'] == target_date].index[0] - 1
    res['y'] = float(df.iloc[prev_idx]['y']) if prev_idx >= 0 else None
    res['y_next'] = float(future_row['y'].iloc[0])
    
    if res['y'] is not None and res['y_next'] is not None:
        res['y_change'] = res['y_next'] - res['y']
        res['direction'] = 1 if res['y_change'] > 0 else 0
        res['predicted_direction'] = 1 if res['yhat'] > res['y'] else 0
        res['actual_direction'] = res['direction']
    else:
        res['y_change'], res['direction'], res['predicted_direction'], res['actual_direction'] = None, None, None, None

    res['Volume'] = int(future_row['Volume'].iloc[0])
    res['EMA'] = float(future_row['EMA'].iloc[0])
    res['corn_close'] = float(future_row['corn_close'].iloc[0]) if not pd.isna(future_row['corn_close'].iloc[0]) else None
    res['wheat_close'] = float(future_row['wheat_close'].iloc[0]) if not pd.isna(future_row['wheat_close'].iloc[0]) else None
    res['Volume_lag1'] = float(future_row['Volume_lag1'].iloc[0])
    res['EMA_lag1'] = float(future_row['EMA_lag1'].iloc[0])
    res['corn_close_lag6'] = float(future_row['corn_close_lag6'].iloc[0]) if use_corn else None
    res['wheat_close_lag1'] = float(future_row['wheat_close_lag1'].iloc[0]) if use_wheat else None
    
    res['used_corn'] = bool(use_corn)
    res['used_wheat'] = bool(use_wheat)
    res['volatility'] = float(res['yhat_upper'] - res['yhat_lower'])
    res['ds'] = res['ds'].strftime('%Y-%m-%d')
    
    return res

def insert_to_bq(**context):
    feature_data = context['ti'].xcom_pull(task_ids='generate_features')
    if not feature_data: return
    client = get_bq_client()
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TARGET_TABLE}"
    row = {k: (None if pd.isna(v) else v) for k, v in feature_data.items()}
    logging.info(f"[PRICE-MODEL][{COMMODITY}] 🔍 적재 예정 데이터:\n{json.dumps(row, indent=2, ensure_ascii=False)}")
    errors = client.insert_rows_json(table_id, [row])
    if errors: raise RuntimeError(f"BQ 적재 실패: {errors}")
    logging.info(f"[PRICE-MODEL][{COMMODITY}] ✅ 적재 완료 성공!")

with DAG(
    'generate_prophet_features_soybean_v1',
    default_args=default_args,
    description='Soybean Prophet 피처 생성 (존재 시 Skip)',
    schedule_interval='40 16 * * *',
    catchup=True,
    max_active_runs=1,
    tags=['soybean', 'prophet', 'feature_engineering']
) as dag:
    t0 = ShortCircuitOperator(task_id='check_exists', python_callable=check_already_exists, provide_context=True)
    t1 = PythonOperator(task_id='generate_features', python_callable=generate_features, provide_context=True)
    t2 = PythonOperator(task_id='insert_to_bq', python_callable=insert_to_bq, provide_context=True)
    t0 >> t1 >> t2