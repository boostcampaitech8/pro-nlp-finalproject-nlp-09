from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import sys
import os
import json

# [환경 설정] 서버 배포 경로 반영
PROJECT_ROOT = "/data/ephemeral/home/jb/pro-nlp-finalproject-nlp-09" 
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    # app 디렉토리 추가
    app_dir = os.path.join(PROJECT_ROOT, "app")
    if app_dir not in sys.path:
        sys.path.append(app_dir)

try:
    from app.routes.orchestrator import run_market_analysis
    from app.utils.data_loader import load_timeseries_prediction
    # 데이터셋 ID는 명시적으로 'tilda' 사용
    DATASET_ID = "tilda"
except ImportError as e:
    print(f"❌ Import Error: {e}")
    run_market_analysis = None
    load_timeseries_prediction = None
    DATASET_ID = "tilda"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'market_prediction_loader_v1',
    default_args=default_args,
    description='시장 예측 분석 및 시계열 데이터 적재 파이프라인 (Tilda Dataset)',
    schedule_interval='@daily',
    catchup=False,
    tags=['market', 'prediction', 'bigquery', 'tilda']
) as dag:

    def analyze_market_task(**context):
        """시장 분석을 수행하고 결과를 XCom에 저장"""
        if not run_market_analysis:
            raise ImportError("run_market_analysis 함수를 불러오지 못했습니다. PROJECT_ROOT를 확인하세요.")
            
        # Airflow 실행 날짜 (YYYY-MM-DD)
        execution_date = context['ds'] 
        
        # [테스트 공지] 현재 데이터 부재 방지를 위해 2025-11-10로 고정하여 실행합니다.
        # 실제 운영 시 target_date=execution_date 로 변경하세요.
        target_date = "2025-11-10" 
        
        print(f"🚀 [Task 1] 시장 분석 시작 (Target: {target_date}, RunDate: {execution_date})")
        result = run_market_analysis(target_date=target_date)
        
        return result

    def load_timeseries_task(**context):
        """XCom에서 데이터를 받아 시계열 테이블에 적재"""
        if not load_timeseries_prediction:
            raise ImportError("load_timeseries_prediction 함수를 불러오지 못했습니다.")

        analysis_result = context['ti'].xcom_pull(task_ids='run_analysis')
        
        if not analysis_result:
            raise ValueError("분석 결과가 없습니다 (XCom Pull Failed).")
            
        timeseries_data = analysis_result.get('timeseries_data')
        if not timeseries_data:
            print("⚠️ 시계열 데이터가 없습니다. 적재를 건너뜁니다.")
            return

        print(f"💾 [Task 2] 시계열 데이터 적재 시작 (Dataset: {DATASET_ID})")
        load_timeseries_prediction(timeseries_data, dataset_id=DATASET_ID)

    # Task 정의
    t1_analyze = PythonOperator(
        task_id='run_analysis',
        python_callable=analyze_market_task,
        provide_context=True
    )

    t2_load_timeseries = PythonOperator(
        task_id='load_timeseries',
        python_callable=load_timeseries_task,
        provide_context=True
    )

    # 실행 순서
    t1_analyze >> t2_load_timeseries