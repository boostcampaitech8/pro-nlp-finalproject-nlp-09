from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from prophet import Prophet
import logging
from google.cloud import bigquery
import os
import sys

# [환경 설정]
PROJECT_ID = os.getenv("VERTEX_AI_PROJECT_ID", "team-blue-448407") # 기본값 설정
DATASET_ID = "tilda"
PRICE_TABLE = "corn_price"
TARGET_TABLE = "prophet_corn"
COMMODITY = "corn"

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
    """
    prophet_corn 테이블의 최신 날짜와 corn_price의 최신 날짜를 비교하여
    업데이트가 필요한지 확인합니다.
    """
    client = get_bq_client()
    
    # 1. 적재된 최신 날짜 확인
    try:
        query_target = f"SELECT MAX(ds) as max_date FROM `{PROJECT_ID}.{DATASET_ID}.{TARGET_TABLE}`"
        df_target = client.query(query_target).to_dataframe()
        last_loaded_date = df_target['max_date'].iloc[0]
    except Exception:
        last_loaded_date = None # 테이블이 없거나 비어있음
        
    # 2. 원본 데이터 최신 날짜 확인
    query_source = f"SELECT MAX(DATE(time)) as max_date FROM `{PROJECT_ID}.{DATASET_ID}.{PRICE_TABLE}`"
    df_source = client.query(query_source).to_dataframe()
    latest_source_date = df_source['max_date'].iloc[0]
    
    if not latest_source_date:
        logging.info("원본 데이터가 없습니다.")
        return False

    logging.info(f"Last Loaded: {last_loaded_date}, Latest Source: {latest_source_date}")
    
    # 업데이트 필요 여부 판단
    if last_loaded_date is None or latest_source_date > last_loaded_date:
        # XCom에 날짜 정보 저장
        context['ti'].xcom_push(key='last_loaded_date', value=str(last_loaded_date) if last_loaded_date else '2020-01-01')
        context['ti'].xcom_push(key='target_date', value=str(latest_source_date))
        return True
    else:
        logging.info("이미 최신 데이터가 적재되어 있습니다.")
        return False

def generate_features(**context):
    """
    Prophet 모델을 학습하고 최신 날짜에 대한 피처를 생성합니다.
    (run_prophet.py 로직 내장)
    """
    # 1. 실행 여부 확인
    is_update_needed = context['ti'].xcom_pull(task_ids='check_new_data')
    if not is_update_needed:
        logging.info("업데이트가 필요 없어 작업을 건너뜁니다.")
        return None

    target_date_str = context['ti'].xcom_pull(key='target_date', task_ids='check_new_data')
    target_date = pd.to_datetime(target_date_str)
    
    client = get_bq_client()
    
    # 2. 학습용 과거 데이터 조회 (최근 4년 + 타겟 날짜)
    # Granger 검증 결과: Corn은 Soybean(lag 8) 필요
    logging.info("데이터 로딩 중...")
    
    start_date = target_date - timedelta(days=365 * 4 + 30) # 넉넉하게 조회
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    # Corn 데이터
    q_corn = f"""
        SELECT time, close, volume as Volume, ema as EMA
        FROM `{PROJECT_ID}.{DATASET_ID}.{PRICE_TABLE}`
        WHERE DATE(time) >= '{start_date_str}' AND DATE(time) <= '{target_date_str}'
        ORDER BY time
    """
    df = client.query(q_corn).to_dataframe()
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

    # 3. 피처 엔지니어링 (Lag 생성)
    # Corn: EMA_lag2, Volume_lag5, soybean_close_lag8
    df['EMA_lag2'] = df['EMA'].shift(2)
    df['Volume_lag5'] = df['Volume'].shift(5)
    df['soybean_close_lag8'] = df['soybean_close'].shift(8)
    
    # 학습에 사용할 데이터 (NaN 제거)
    # 주의: 타겟 날짜(맨 마지막 행)는 예측해야 하므로 남겨야 함
    # 맨 마지막 행은 y(오늘 종가)를 모르고 예측해야 하는 시점(내일)이 아니라,
    # '오늘' 마감된 데이터를 바탕으로 '오늘의 지표'를 기록하는 것임.
    # 하지만 Prophet 학습 시에는 y가 있어야 함.
    
    # 전략: 
    # 1. target_date 이전 데이터로 학습
    # 2. target_date(오늘)를 예측
    
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
    
    # 4. Prophet 모델 학습
    model = Prophet(
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    for reg in regressors:
        model.add_regressor(reg, mode='multiplicative')
        
    model.fit(train_df[['ds', 'y'] + regressors])
    
    # 5. 예측 (Target Date 1건)
    future_row = df[df['ds'] == target_date].copy()
    if future_row.empty:
        raise ValueError(f"{target_date} 데이터가 데이터프레임에 없습니다.")
        
    # 필요한 Feature가 다 있는지 확인
    if future_row[regressors].isna().any().any():
        # Feature가 없으면 이전 값으로 채우거나(Forward Fill) 에러 처리
        # 여기서는 안전하게 이전 값 사용 시도
        logging.warning("Target date의 Feature에 결측치가 있어 전일 값으로 대체합니다.")
        future_row[regressors] = future_row[regressors].fillna(method='ffill')
        
    forecast = model.predict(future_row)
    
    # 6. 결과 정리
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].iloc[0].to_dict()
    
    # 컴포넌트 추가
    for col in ['weekly', 'yearly', 'extra_regressors_multiplicative']:
        if col in forecast.columns:
            result[col] = forecast[col].iloc[0]
            
    for reg in regressors:
        if reg in forecast.columns:
            result[f"{reg}_effect"] = forecast[reg].iloc[0]
            
    # 원본 데이터 추가
    result['y'] = future_row['y'].iloc[0] # 이게 '오늘 종가'
    # y_next는 '내일 종가'인데 아직 모름 (None) -> 나중에 채워짐
    result['y_next'] = None 
    
    result['Volume'] = future_row['Volume'].iloc[0]
    result['EMA'] = future_row['EMA'].iloc[0]
    result['corn_close'] = future_row['close'].iloc[0]
    result['soybean_close'] = future_row['soybean_close'].iloc[0]
    
    # Lag 값들
    result['EMA_lag2'] = future_row['EMA_lag2'].iloc[0]
    result['Volume_lag5'] = future_row['Volume_lag5'].iloc[0]
    if use_soybean:
        result['soybean_close_lag8'] = future_row['soybean_close_lag8'].iloc[0]
    
    result['used_soybean'] = use_soybean
    result['used_corn'] = True # 자기 자신
    
    # 변동성 등 파생지표
    result['volatility'] = result['yhat_upper'] - result['yhat_lower']
    
    # 날짜 포맷 변환 (datetime -> str)
    result['ds'] = result['ds'].strftime('%Y-%m-%d')
    
    logging.info(f"생성된 피처: {result}")
    return result

def insert_to_bq(**context):
    """생성된 피처를 BigQuery에 적재"""
    feature_data = context['ti'].xcom_pull(task_ids='generate_features')
    
    if not feature_data:
        logging.info("적재할 데이터가 없습니다.")
        return

    client = get_bq_client()
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TARGET_TABLE}"
    
    # JSON 적재를 위해 데이터 타입 보정
    # NaN -> None 변환
    row = {k: (None if pd.isna(v) else v) for k, v in feature_data.items()}
    
    errors = client.insert_rows_json(table_id, [row])
    
    if errors:
        raise RuntimeError(f"BigQuery 적재 실패: {errors}")
        
    logging.info(f"적재 완료: {row['ds']}")

with DAG(
    'generate_prophet_features_corn_v1',
    default_args=default_args,
    description='Corn 가격 데이터를 가공하여 Prophet 피처 생성 및 적재 (Incremental)',
    schedule_interval='30 16 * * *', # 매일 16:30
    catchup=False,
    tags=['corn', 'prophet', 'feature_engineering']
) as dag:

    t1 = PythonOperator(
        task_id='check_new_data',
        python_callable=check_new_data,
        provide_context=True
    )

    t2 = PythonOperator(
        task_id='generate_features',
        python_callable=generate_features,
        provide_context=True
    )

    t3 = PythonOperator(
        task_id='insert_to_bq',
        python_callable=insert_to_bq,
        provide_context=True
    )

    t1 >> t2 >> t3
