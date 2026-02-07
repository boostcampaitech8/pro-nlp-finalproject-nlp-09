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
PRICE_TABLE = "corn_price"
TARGET_TABLE = "prophet_corn"
COMMODITY = "corn"
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
    (prophet_inference.py 로직과 100% 일치화)
    """
    target_date_str = context['ds']
    target_date = pd.to_datetime(target_date_str)
    
    logging.info(f"[PRICE-MODEL][{COMMODITY}] 🚀 Task 시작: generate_features (Target: {target_date_str})")
    
    client = get_bq_client()
    start_date = target_date - timedelta(days=365 * 4 + 30)
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    # 1. Corn 데이터 로드
    q_corn = f"SELECT time, close, open, high, low, volume as Volume, ema as EMA FROM `{PROJECT_ID}.{DATASET_ID}.{PRICE_TABLE}` WHERE DATE(time) >= '{start_date_str}' AND DATE(time) <= '{target_date_str}' ORDER BY time"
    df = client.query(q_corn).to_dataframe()
    
    if df.empty or df.iloc[-1]['time'].strftime('%Y-%m-%d') != target_date_str:
        logging.warning(f"[PRICE-MODEL][{COMMODITY}] ⚠️ {target_date_str} 데이터 부재로 중단.")
        return None

    df['ds'] = pd.to_datetime(df['time'])
    df['y'] = pd.to_numeric(df['close'])
    
    # 2. 보조 데이터(Soybean) 로드 (Lag 8용)
    q_soy = f"SELECT time, close as soybean_close FROM `{PROJECT_ID}.{DATASET_ID}.soybean_price` WHERE DATE(time) >= '{start_date_str}' AND DATE(time) <= '{target_date_str}' ORDER BY time"
    try:
        df_soy = client.query(q_soy).to_dataframe()
        df_soy['time'] = pd.to_datetime(df_soy['time'])
        df = pd.merge(df, df_soy, on='time', how='left')
    except Exception: df['soybean_close'] = np.nan

    # 3. Granger Lag 피처 생성
    df['EMA_lag2'] = df['EMA'].shift(2).astype(float)
    df['Volume_lag5'] = df['Volume'].shift(5).astype(float)
    df['soybean_close_lag8'] = df['soybean_close'].shift(8).astype(float)
    
    # 학습 데이터 준비 (target_date 전날까지)
    train_df = df[df['ds'] < target_date].dropna(subset=['EMA_lag2', 'Volume_lag5']).copy()
    
    # Soybean 사용 여부 결정
    use_soybean = False
    if 'soybean_close_lag8' in train_df.columns:
        if train_df['soybean_close_lag8'].notna().mean() > 0.5:
            use_soybean = True
            train_df = train_df.dropna(subset=['soybean_close_lag8'])
    
    regressors = ['EMA_lag2', 'Volume_lag5']
    if use_soybean: regressors.append('soybean_close_lag8')
    
    # 4. Prophet 모델 학습
    model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    for reg in regressors: model.add_regressor(reg, mode='multiplicative')
    model.fit(train_df[['ds', 'y'] + regressors])
    
    # 5. 예측 (Target Date 1건)
    future_row = df[df['ds'] == target_date].copy()
    if future_row[regressors].isna().any().any():
        future_row[regressors] = future_row[regressors].fillna(method='ffill')
        
    forecast = model.predict(future_row)
    
    # 6. 결과 딕셔너리 생성 (팀원 스키마 완벽 대응)
    res = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].iloc[0].to_dict()
    
    # 성분 추가
    for col in ['weekly', 'yearly', 'extra_regressors_multiplicative']:
        res[col] = forecast[col].iloc[0] if col in forecast.columns else None
            
    # Effect 컬럼 추가
    for reg in regressors:
        res[f"{reg}_effect"] = forecast[reg].iloc[0] if reg in forecast.columns else None
            
    # 가격 데이터 매핑 (가장 중요!)
    # 팀원 로직: y = 전날 종가, y_next = 오늘 종가
    prev_idx = df[df['ds'] == target_date].index[0] - 1
    res['y'] = float(df.iloc[prev_idx]['y']) if prev_idx >= 0 else None
    res['y_next'] = float(future_row['y'].iloc[0]) # 오늘 종가
    
    # 파생 변수 계산
    if res['y'] is not None and res['y_next'] is not None:
        res['y_change'] = res['y_next'] - res['y']
        res['direction'] = 1 if res['y_change'] > 0 else 0
        res['predicted_direction'] = 1 if res['yhat'] > res['y'] else 0
        res['actual_direction'] = res['direction']
    else:
        res['y_change'], res['direction'], res['predicted_direction'], res['actual_direction'] = None, None, None, None

    # 기타 지표
    res['Volume'] = int(future_row['Volume'].iloc[0])
    res['EMA'] = float(future_row['EMA'].iloc[0])
    res['corn_close'] = float(future_row['close'].iloc[0])
    res['soybean_close'] = float(future_row['soybean_close'].iloc[0]) if not pd.isna(future_row['soybean_close'].iloc[0]) else None
    res['EMA_lag2'] = float(future_row['EMA_lag2'].iloc[0])
    res['Volume_lag5'] = float(future_row['Volume_lag5'].iloc[0])
    res['soybean_close_lag8'] = float(future_row['soybean_close_lag8'].iloc[0]) if use_soybean else None
    
    res['used_soybean'] = bool(use_soybean)
    res['volatility'] = float(res['yhat_upper'] - res['yhat_lower'])
    res['ds'] = res['ds'].strftime('%Y-%m-%d')
    
    logging.info(f"[PRICE-MODEL][{COMMODITY}] ✨ 피처 생성 완료: {target_date_str}")
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
    'generate_prophet_features_corn_v1',
    default_args=default_args,
    description='Corn Prophet 피처 생성 (존재 시 Skip)',
    schedule_interval='30 16 * * *',
    catchup=True,
    max_active_runs=1,
    tags=['corn', 'prophet', 'feature_engineering']
) as dag:

    t0 = ShortCircuitOperator(task_id='check_exists', python_callable=check_already_exists, provide_context=True)
    t1 = PythonOperator(task_id='generate_features', python_callable=generate_features, provide_context=True)
    t2 = PythonOperator(task_id='insert_to_bq', python_callable=insert_to_bq, provide_context=True)

    t0 >> t1 >> t2
