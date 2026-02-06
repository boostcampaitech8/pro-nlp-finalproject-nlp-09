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
PRICE_TABLE = "wheat_price"
TARGET_TABLE = "prophet_wheat"
COMMODITY = "wheat"
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
    logging.info(f"[PRICE-MODEL][{COMMODITY}] 🚀 Task 시작: generate_features (Target: {target_date_str})")
    
    client = get_bq_client()
    start_date = target_date - timedelta(days=365 * 4 + 30)
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    logging.info(f"[PRICE-MODEL][{COMMODITY}] 🔍 BQ 데이터 조회 범위: {start_date_str} ~ {target_date_str}")
    
    # 1. Wheat 데이터
    q_wheat = f"SELECT time, close, volume as Volume, ema as EMA FROM `{PROJECT_ID}.{DATASET_ID}.{PRICE_TABLE}` WHERE DATE(time) >= '{start_date_str}' AND DATE(time) <= '{target_date_str}' ORDER BY time"
    df = client.query(q_wheat).to_dataframe()
    
    if df.empty or df.iloc[-1]['time'].strftime('%Y-%m-%d') != target_date_str:
        logging.warning(f"[PRICE-MODEL][{COMMODITY}] ⚠️ {target_date_str} 가격 데이터가 아직 없습니다. 작업을 건너뜁니다.")
        return None

    logging.info(f"[PRICE-MODEL][{COMMODITY}] ✅ {PRICE_TABLE} 데이터 로드 성공: {len(df)}건")
    df['ds'] = pd.to_datetime(df['time'])
    df['y'] = pd.to_numeric(df['close'])
    
    # 2. Corn & Soybean 데이터
    for dep in ['corn', 'soybean']:
        q_dep = f"SELECT time, close as {dep}_close FROM `{PROJECT_ID}.{DATASET_ID}.{dep}_price` WHERE DATE(time) >= '{start_date_str}' AND DATE(time) <= '{target_date_str}' ORDER BY time"
        try:
            df_dep = client.query(q_dep).to_dataframe()
            df_dep['time'] = pd.to_datetime(df_dep['time'])
            df = pd.merge(df, df_dep, on='time', how='left')
            logging.info(f"[PRICE-MODEL][{COMMODITY}] ✅ 보조 데이터({dep}) 병합 완료: {len(df_dep)}건")
        except Exception as e:
            logging.warning(f"[PRICE-MODEL][{COMMODITY}] ⚠️ {dep} 데이터 로드 실패: {e}")
            df[f'{dep}_close'] = np.nan

    # 3. Lag 생성
    df['EMA_lag1'] = df['EMA'].shift(1)
    df['Volume_lag1'] = df['Volume'].shift(1)
    df['corn_close_lag2'] = df['corn_close'].shift(2)
    df['soybean_close_lag1'] = df['soybean_close'].shift(1)
    
    train_df = df[df['ds'] < target_date].dropna(subset=['EMA_lag1', 'Volume_lag1']).copy()
    
    base_regressors = ['EMA_lag1', 'Volume_lag1']
    extra_regressors = []
    if 'corn_close_lag2' in train_df.columns and train_df['corn_close_lag2'].notna().mean() > 0.5:
        extra_regressors.append('corn_close_lag2')
    if 'soybean_close_lag1' in train_df.columns and train_df['soybean_close_lag1'].notna().mean() > 0.5:
        extra_regressors.append('soybean_close_lag1')
        
    regressors = base_regressors + extra_regressors
    logging.info(f"[PRICE-MODEL][{COMMODITY}] 🧠 학습 데이터 준비 완료: {len(train_df)}행, Regressors: {regressors}")
    
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
    result['soybean_close'] = future_row['soybean_close'].iloc[0] if 'soybean_close' in future_row else None
    result['EMA_lag1'] = future_row['EMA_lag1'].iloc[0]
    result['Volume_lag1'] = future_row['Volume_lag1'].iloc[0]
    result['corn_close_lag2'] = future_row['corn_close_lag2'].iloc[0] if 'corn_close_lag2' in future_row else None
    result['soybean_close_lag1'] = future_row['soybean_close_lag1'].iloc[0] if 'soybean_close_lag1' in future_row else None
    result['used_corn'] = 'corn_close_lag2' in extra_regressors
    result['used_soybean'] = 'soybean_close_lag1' in extra_regressors
    result['volatility'] = result['yhat_upper'] - result['yhat_lower']
    result['ds'] = result['ds'].strftime('%Y-%m-%d')
    
    logging.info(f"[PRICE-MODEL][{COMMODITY}] ✨ 피처 생성 성공: yhat={result['yhat']:.2f}, trend={result['trend']:.2f}")
    return result

def insert_to_bq(**context):
    feature_data = context['ti'].xcom_pull(task_ids='generate_features')
    if not feature_data:
        logging.info(f"[PRICE-MODEL][{COMMODITY}] ⏹️ 적재할 데이터가 없습니다.")
        return
    client = get_bq_client()
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TARGET_TABLE}"
    target_ds = feature_data['ds']
    
    # [상세 로깅] 적재될 데이터 전체 출력
    row = {k: (None if pd.isna(v) else v) for k, v in feature_data.items()}
    logging.info(f"[PRICE-MODEL][{COMMODITY}] 🔍 적재 예정 컬럼 ({len(row)}개): {list(row.keys())}")
    logging.info(f"[PRICE-MODEL][{COMMODITY}] 📄 적재 예정 데이터 상세:\n{json.dumps(row, indent=2, ensure_ascii=False)}")

    logging.info(f"[PRICE-MODEL][{COMMODITY}] 💾 BQ 적재 시작 (Target: {target_ds})")
    try:
        client.query(f"DELETE FROM `{table_id}` WHERE ds = '{target_ds}'").result()
        logging.info(f"[PRICE-MODEL][{COMMODITY}] 🗑️ 기존 데이터 삭제 완료 ({target_ds})")
    except Exception: pass

    errors = client.insert_rows_json(table_id, [row])
    if errors: raise RuntimeError(f"BigQuery 적재 실패: {errors}")
    logging.info(f"[PRICE-MODEL][{COMMODITY}] ✅ 적재 완료 성공!")

with DAG(
    'generate_prophet_features_wheat_v1',
    default_args=default_args,
    description='Wheat Prophet 피처 생성 (Backfill 지원)',
    schedule_interval='50 16 * * *',
    catchup=True,
    max_active_runs=1,
    tags=['wheat', 'prophet', 'feature_engineering']
) as dag:

    t1 = PythonOperator(task_id='generate_features', python_callable=generate_features, provide_context=True)
    t2 = PythonOperator(task_id='insert_to_bq', python_callable=insert_to_bq, provide_context=True)
    t1 >> t2