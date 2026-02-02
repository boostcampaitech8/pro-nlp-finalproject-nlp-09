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
    from app.utils.data_loader import load_timeseries_prediction, load_news_prediction, upload_report_to_gcs
    from app.config.settings import BIGQUERY_DATASET_ID
except ImportError as e:
    print(f"❌ Import Error: {e}")
    run_market_analysis = None
    load_timeseries_prediction = None
    load_news_prediction = None
    upload_report_to_gcs = None
    BIGQUERY_DATASET_ID = "tilda"

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
            
        execution_date = context['ds'] 
        # [테스트 공지] 현재 데이터 부재 방지를 위해 2025-11-10로 고정하여 실행합니다.
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

        print(f"💾 [Task 2] 시계열 데이터 적재 시작 (Dataset: {BIGQUERY_DATASET_ID})")
        load_timeseries_prediction(timeseries_data, dataset_id=BIGQUERY_DATASET_ID)

    def load_news_task(**context):
        """XCom에서 데이터를 받아 뉴스 테이블에 적재"""
        if not load_news_prediction:
            raise ImportError("load_news_prediction 함수를 불러오지 못했습니다.")

        analysis_result = context['ti'].xcom_pull(task_ids='run_analysis')
        if not analysis_result:
            raise ValueError("분석 결과가 없습니다.")
            
        news_data = analysis_result.get('news_data')
        if not news_data:
            print("⚠️ 뉴스 예측 데이터가 없습니다. 적재를 건너뜁니다.")
            return

        print(f"💾 [Task 3] 뉴스 데이터 적재 시작 (Dataset: {BIGQUERY_DATASET_ID})")
        load_news_prediction(news_data, dataset_id=BIGQUERY_DATASET_ID)

    def upload_report_task(**context):
        """XCom에서 리포트를 받아 GCS에 업로드"""
        if not upload_report_to_gcs:
            raise ImportError("upload_report_to_gcs 함수를 불러오지 못했습니다.")

        analysis_result = context['ti'].xcom_pull(task_ids='run_analysis')
        if not analysis_result:
            raise ValueError("분석 결과가 없습니다.")
            
        final_report = analysis_result.get('final_report')
        target_date = analysis_result.get('target_date')
        
        if not final_report:
            print("⚠️ 최종 리포트가 없습니다. 업로드를 건너뜁니다.")
            return

        BUCKET_NAME = "agri-market-reports" 
        print(f"☁️ [Task 4] 리포트 GCS 업로드 시작 (Bucket: {BUCKET_NAME})")
        upload_report_to_gcs(final_report, target_date, bucket_name=BUCKET_NAME)

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

    t3_load_news = PythonOperator(
        task_id='load_news',
        python_callable=load_news_task,
        provide_context=True
    )

    t4_upload_report = PythonOperator(
        task_id='upload_report',
        python_callable=upload_report_task,
        provide_context=True
    )

    # 실행 순서: 분석 -> [시계열 적재, 뉴스 적재, 리포트 업로드] 병렬 실행
    t1_analyze >> [t2_load_timeseries, t3_load_news, t4_upload_report]