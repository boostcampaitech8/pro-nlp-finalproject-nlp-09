from airflow import DAG
from airflow.operators.python import PythonOperator
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

def generate_features(**context):
    """
    Airflow 실행 날짜(ds)를 기준으로 Prophet 모델을 학습하고 피처를 생성합니다.
    """
    target_date_str = context['ds']
    target_date = pd.to_datetime(target_date_str)
    
    logging.info(f"[PRICE-MODEL][{COMMODITY}] 🚀 Task 시작: generate_features (Target: {target_date_str})")
    
    client = get_bq_client()
    
    # 1. 학습용 과거 데이터 조회 (타겟 날짜 기준 과거 4년 + 타겟 날짜 포함)
    start_date = target_date - timedelta(days=365 * 4 + 30)
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    logging.info(f"[PRICE-MODEL][{COMMODITY}] 🔍 BQ 데이터 조회 범위: {start_date_str} ~ {target_date_str}")
    
    # Corn 데이터
    q_corn = f"""
        SELECT time, close, volume as Volume, ema as EMA
        FROM `{PROJECT_ID}.{DATASET_ID}.{PRICE_TABLE}`
        WHERE DATE(time) >= '{start_date_str}' AND DATE(time) <= '{target_date_str}'
        ORDER BY time
    """
    df = client.query(q_corn).to_dataframe()
    
    if df.empty or df.iloc[-1]['time'].strftime('%Y-%m-%d') != target_date_str:
        logging.warning(f"[PRICE-MODEL][{COMMODITY}] ⚠️ {target_date_str} 가격 데이터가 아직 없습니다. 작업을 건너뜁니다.")
        return None

    logging.info(f"[PRICE-MODEL][{COMMODITY}] ✅ {PRICE_TABLE} 데이터 로드 성공: {len(df)}건")

    df['ds'] = pd.to_datetime(df['time'])
    df['y'] = pd.to_numeric(df['close'])
    
    # Soybean 데이터
    q_soy = f"""
        SELECT time, close as soybean_close
        FROM `{PROJECT_ID}.{DATASET_ID}.soybean_price`
        WHERE DATE(time) >= '{start_date_str}' AND DATE(time) <= '{target_date_str}'
        ORDER BY time
    """
    try:
        df_soy = client.query(q_soy).to_dataframe()
        df_soy['time'] = pd.to_datetime(df_soy['time'])
        df = pd.merge(df, df_soy, on='time', how='left')
        logging.info(f"[PRICE-MODEL][{COMMODITY}] ✅ 보조 데이터(Soybean) 병합 완료: {len(df_soy)}건")
    except Exception as e:
        logging.warning(f"[PRICE-MODEL][{COMMODITY}] ⚠️ Soybean 데이터 로드 실패: {e}")
        df['soybean_close'] = np.nan

    # 2. 피처 엔지니어링 (Lag 생성)
    df['EMA_lag2'] = df['EMA'].shift(2)
    df['Volume_lag5'] = df['Volume'].shift(5)
    df['soybean_close_lag8'] = df['soybean_close'].shift(8)
    
    train_df = df[df['ds'] < target_date].dropna(subset=['EMA_lag2', 'Volume_lag5']).copy()
    
    use_soybean = False
    if 'soybean_close_lag8' in train_df.columns:
        if train_df['soybean_close_lag8'].notna().mean() > 0.5:
            use_soybean = True
            train_df = train_df.dropna(subset=['soybean_close_lag8'])
    
    regressors = ['EMA_lag2', 'Volume_lag5']
    if use_soybean: regressors.append('soybean_close_lag8')
        
    logging.info(f"[PRICE-MODEL][{COMMODITY}] 🧠 학습 데이터 준비 완료: {len(train_df)}행, Regressors: {regressors}")
    
    model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    for reg in regressors: model.add_regressor(reg, mode='multiplicative')
    model.fit(train_df[['ds', 'y'] + regressors])
    
    # 4. 예측
    future_row = df[df['ds'] == target_date].copy()
    if future_row[regressors].isna().any().any():
        future_row[regressors] = future_row[regressors].fillna(method='ffill')
        
    forecast = model.predict(future_row)
    
    # 5. 결과 정리
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].iloc[0].to_dict()
    for col in ['weekly', 'yearly', 'extra_regressors_multiplicative']:
        if col in forecast.columns: result[col] = forecast[col].iloc[0]
    for reg in regressors:
        if reg in forecast.columns: result[f"{reg}_effect"] = forecast[reg].iloc[0]
            
    result['y'] = future_row['y'].iloc[0] 
    result['y_next'] = None
    result['Volume'] = future_row['Volume'].iloc[0]
    result['EMA'] = future_row['EMA'].iloc[0]
    result['corn_close'] = future_row['close'].iloc[0]
    result['soybean_close'] = future_row['soybean_close'].iloc[0]
    result['EMA_lag2'] = future_row['EMA_lag2'].iloc[0]
    result['Volume_lag5'] = future_row['Volume_lag5'].iloc[0]
    if use_soybean: result['soybean_close_lag8'] = future_row['soybean_close_lag8'].iloc[0]
    
    result['used_soybean'] = use_soybean
    result['used_corn'] = True
    result['volatility'] = result['yhat_upper'] - result['yhat_lower']
    result['ds'] = result['ds'].strftime('%Y-%m-%d')
    
    logging.info(f"[PRICE-MODEL][{COMMODITY}] ✨ 피처 생성 성공: yhat={result['yhat']:.2f}, trend={result['trend']:.2f}")
    return result

def insert_to_bq(**context):
    """생성된 피처를 BigQuery에 적재"""
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
    except Exception as e:
        logging.warning(f"[PRICE-MODEL][{COMMODITY}] ⚠️ 삭제 쿼리 실패: {e}")

    errors = client.insert_rows_json(table_id, [row])
    
    if errors:
        raise RuntimeError(f"BigQuery 적재 실패: {errors}")
        
    logging.info(f"[PRICE-MODEL][{COMMODITY}] ✅ 적재 완료 성공!")

with DAG(
    'generate_prophet_features_corn_v1',
    default_args=default_args,
    description='Corn Prophet 피처 생성 (Backfill 지원)',
    schedule_interval='30 16 * * *',
    catchup=True,
    max_active_runs=1,
    tags=['corn', 'prophet', 'feature_engineering']
) as dag:

    t1 = PythonOperator(task_id='generate_features', python_callable=generate_features, provide_context=True)
    t2 = PythonOperator(task_id='insert_to_bq', python_callable=insert_to_bq, provide_context=True)
    t1 >> t2
