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
PROJECT_ID = os.getenv("VERTEX_AI_PROJECT_ID", "project-5b75bb04-485d-454e-af7")
DATASET_ID = "tilda"
PRICE_TABLE = "corn_price"
TARGET_TABLE = "prophet_corn"
COMMODITY = "corn"

# 시작 날짜: 적재를 시작하고 싶은 과거 날짜로 설정 (예: 2024년 1월 1일)
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
    # Airflow가 지정해준 실행 기준 날짜 (YYYY-MM-DD)
    target_date_str = context['ds']
    target_date = pd.to_datetime(target_date_str)
    
    logging.info(f"Target Date: {target_date_str}에 대한 피처 생성을 시작합니다.")
    
    client = get_bq_client()
    
    # 1. 학습용 과거 데이터 조회 (타겟 날짜 기준 과거 4년 + 타겟 날짜 포함)
    start_date = target_date - timedelta(days=365 * 4 + 30)
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    # Corn 데이터
    q_corn = f"""
        SELECT time, close, volume as Volume, ema as EMA
        FROM `{PROJECT_ID}.{DATASET_ID}.{PRICE_TABLE}`
        WHERE DATE(time) >= '{start_date_str}' AND DATE(time) <= '{target_date_str}'
        ORDER BY time
    """
    df = client.query(q_corn).to_dataframe()
    
    # 타겟 날짜의 데이터가 아직 없을 수도 있음 (미래 시점 예측인 경우 등)
    # 하지만 피처 생성을 위해서는 최소한 타겟 날짜의 데이터가 있어야 'y_next' 등을 제외한 나머지 값들을 채울 수 있음
    if df.empty or df.iloc[-1]['time'].strftime('%Y-%m-%d') != target_date_str:
        logging.warning(f"{target_date_str}의 가격 데이터가 아직 {PRICE_TABLE}에 없습니다. 작업을 건너뜁니다.")
        return None

    df['ds'] = pd.to_datetime(df['time'])
    df['y'] = pd.to_numeric(df['close'])
    
    # Soybean 데이터 (Lag 8을 위해 필요)
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
    except Exception as e:
        logging.warning(f"Soybean 데이터 로드 실패: {e}")
        df['soybean_close'] = np.nan

    # 2. 피처 엔지니어링 (Lag 생성)
    # Corn: EMA_lag2, Volume_lag5, soybean_close_lag8
    df['EMA_lag2'] = df['EMA'].shift(2)
    df['Volume_lag5'] = df['Volume'].shift(5)
    df['soybean_close_lag8'] = df['soybean_close'].shift(8)
    
    # 학습 데이터 준비 (타겟 날짜 제외)
    train_df = df[df['ds'] < target_date].dropna(subset=['EMA_lag2', 'Volume_lag5']).copy()
    
    # Soybean 데이터가 충분한지 확인
    use_soybean = False
    if 'soybean_close_lag8' in train_df.columns:
        if train_df['soybean_close_lag8'].notna().mean() > 0.5:
            use_soybean = True
            train_df = train_df.dropna(subset=['soybean_close_lag8'])
    
    # Regressors 설정
    regressors = ['EMA_lag2', 'Volume_lag5']
    if use_soybean:
        regressors.append('soybean_close_lag8')
        
    logging.info(f"학습 데이터: {len(train_df)}행, Regressors: {regressors}")
    
    # 3. Prophet 모델 학습
    model = Prophet(
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    for reg in regressors:
        model.add_regressor(reg, mode='multiplicative')
        
    model.fit(train_df[['ds', 'y'] + regressors])
    
    # 4. 예측 (Target Date 1건)
    future_row = df[df['ds'] == target_date].copy()
    
    # Feature 결측치 처리 (Forward Fill)
    if future_row[regressors].isna().any().any():
        future_row[regressors] = future_row[regressors].fillna(method='ffill')
        
    forecast = model.predict(future_row)
    
    # 5. 결과 정리
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].iloc[0].to_dict()
    
    for col in ['weekly', 'yearly', 'extra_regressors_multiplicative']:
        if col in forecast.columns:
            result[col] = forecast[col].iloc[0]
            
    for reg in regressors:
        if reg in forecast.columns:
            result[f"{reg}_effect"] = forecast[reg].iloc[0]
            
    # 원본 데이터 매핑
    result['y'] = future_row['y'].iloc[0] 
    result['y_next'] = None # 실시간 적재 시점에는 알 수 없음
    
    result['Volume'] = future_row['Volume'].iloc[0]
    result['EMA'] = future_row['EMA'].iloc[0]
    result['corn_close'] = future_row['close'].iloc[0]
    result['soybean_close'] = future_row['soybean_close'].iloc[0]
    
    result['EMA_lag2'] = future_row['EMA_lag2'].iloc[0]
    result['Volume_lag5'] = future_row['Volume_lag5'].iloc[0]
    if use_soybean:
        result['soybean_close_lag8'] = future_row['soybean_close_lag8'].iloc[0]
    
    result['used_soybean'] = use_soybean
    result['used_corn'] = True
    result['volatility'] = result['yhat_upper'] - result['yhat_lower']
    result['ds'] = result['ds'].strftime('%Y-%m-%d')
    
    logging.info(f"피처 생성 완료: {result}")
    return result

def insert_to_bq(**context):
    """생성된 피처를 BigQuery에 적재 (중복 시 덮어쓰기 로직 고려 필요)"""
    feature_data = context['ti'].xcom_pull(task_ids='generate_features')
    
    if not feature_data:
        logging.info("적재할 데이터가 없습니다.")
        return

    client = get_bq_client()
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TARGET_TABLE}"
    
    # 기존 데이터 삭제 후 삽입 (DELETE -> INSERT로 중복 방지 구현)
    # MERGE를 쓰면 좋지만 Python Client에서는 쿼리로 직접 날려야 함.
    # 여기서는 간단하게 해당 날짜 데이터가 있으면 지우고 넣는 방식을 사용.
    target_ds = feature_data['ds']
    delete_query = f"DELETE FROM `{table_id}` WHERE ds = '{target_ds}'"
    try:
        client.query(delete_query).result()
        logging.info(f"{target_ds} 데이터 삭제 완료 (덮어쓰기 준비)")
    except Exception as e:
        logging.warning(f"삭제 쿼리 실패 (테이블이 없을 수 있음): {e}")

    # 데이터 삽입
    row = {k: (None if pd.isna(v) else v) for k, v in feature_data.items()}
    errors = client.insert_rows_json(table_id, [row])
    
    if errors:
        raise RuntimeError(f"BigQuery 적재 실패: {errors}")
        
    logging.info(f"적재 완료: {row['ds']}")

with DAG(
    'generate_prophet_features_corn_v1',
    default_args=default_args,
    description='Corn Prophet 피처 생성 (Backfill 지원)',
    schedule_interval='30 16 * * *', # 매일 16:30
    catchup=True, # 과거 데이터 자동 백필 활성화
    max_active_runs=1, # 순차 실행 보장 (DB 부하 방지)
    tags=['corn', 'prophet', 'feature_engineering']
) as dag:

    t1 = PythonOperator(
        task_id='generate_features',
        python_callable=generate_features,
        provide_context=True
    )

    t2 = PythonOperator(
        task_id='insert_to_bq',
        python_callable=insert_to_bq,
        provide_context=True
    )

    t1 >> t2