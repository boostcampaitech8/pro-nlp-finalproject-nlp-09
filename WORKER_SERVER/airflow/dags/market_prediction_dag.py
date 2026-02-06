from airflow import DAG
from airflow.operators.python import PythonOperator
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

# 분석 대상 품목 리스트
COMMODITIES = ["corn", "soybean", "wheat"]

def create_dag(commodity_name):
    """
    품목별 DAG를 동적으로 생성하는 함수
    """
    default_args = {
        'owner': 'airflow',
        'depends_on_past': False,
        'start_date': datetime(2025, 1, 1),
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    }

    dag_id = f'market_prediction_{commodity_name}_v1'
    
    dag = DAG(
        dag_id,
        default_args=default_args,
        description=f'{commodity_name} 시장 예측 분석 및 데이터 적재 파이프라인',
        schedule_interval='0 0 * * 1-5',  # 평일 자정 실행
        max_active_runs=1,
        catchup=True,
        tags=['market', 'prediction', 'bigquery', 'tilda', commodity_name]
    )

    with dag:
        def analyze_market_task(**context):
            """시장 분석 수행"""
            if not run_market_analysis:
                raise ImportError("run_market_analysis 함수를 불러오지 못했습니다.")
                
            execution_date = context['ds'] 
            # [테스트 공지] 현재 데이터 부재 방지를 위해 2025-11-10로 고정하여 실행합니다.
            target_date = "2025-11-10" 
            
            print(f"🚀 [{commodity_name}] 시장 분석 시작 (Target: {target_date}, RunDate: {execution_date})")
            # commodity 인자 전달
            result = run_market_analysis(target_date=target_date, commodity=commodity_name)
            return result

        def load_timeseries_task(**context):
            """시계열 데이터 적재"""
            if not load_timeseries_prediction:
                raise ImportError("load_timeseries_prediction 함수를 불러오지 못했습니다.")

            analysis_result = context['ti'].xcom_pull(task_ids='run_analysis')
            if not analysis_result: raise ValueError("분석 결과가 없습니다.")
                
            timeseries_data = analysis_result.get('timeseries_data')
            if not timeseries_data:
                print(f"⚠️ [{commodity_name}] 시계열 데이터가 없습니다. 적재를 건너뜁니다.")
                return

            print(f"💾 [{commodity_name}] 시계열 데이터 적재 시작 (Dataset: {BIGQUERY_DATASET_ID})")
            # commodity 인자 전달
            load_timeseries_prediction(timeseries_data, commodity=commodity_name, dataset_id=BIGQUERY_DATASET_ID)

        def load_news_task(**context):
            """뉴스 데이터 적재"""
            if not load_news_prediction:
                raise ImportError("load_news_prediction 함수를 불러오지 못했습니다.")

            analysis_result = context['ti'].xcom_pull(task_ids='run_analysis')
            if not analysis_result: raise ValueError("분석 결과가 없습니다.")
                
            news_data = analysis_result.get('news_data')
            if not news_data:
                print(f"⚠️ [{commodity_name}] 뉴스 예측 데이터가 없습니다. 적재를 건너뜁니다.")
                return

            print(f"💾 [{commodity_name}] 뉴스 데이터 적재 시작")
            # commodity 인자 전달
            load_news_prediction(news_data, commodity=commodity_name, dataset_id=BIGQUERY_DATASET_ID)

        def upload_report_task(**context):
            """리포트 GCS 업로드"""
            if not upload_report_to_gcs:
                raise ImportError("upload_report_to_gcs 함수를 불러오지 못했습니다.")

            analysis_result = context['ti'].xcom_pull(task_ids='run_analysis')
            if not analysis_result: raise ValueError("분석 결과가 없습니다.")
                
            final_report = analysis_result.get('final_report')
            target_date = analysis_result.get('target_date')
            
            if not final_report:
                print(f"⚠️ [{commodity_name}] 최종 리포트가 없습니다. 업로드를 건너뜁니다.")
                return

            BUCKET_NAME = "team-blue-raw-data" 
            print(f"☁️ [{commodity_name}] 리포트 GCS 업로드 시작")
            # commodity 인자 전달
            upload_report_to_gcs(final_report, target_date, commodity=commodity_name, bucket_name=BUCKET_NAME)

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

        # 실행 순서
        t1_analyze >> [t2_load_timeseries, t3_load_news, t4_upload_report]

    return dag

# 동적 DAG 생성 (Global Scope에 DAG 객체가 노출되어야 Airflow가 인식함)
for commodity in COMMODITIES:
    dag_id = f'market_prediction_{commodity}_v1'
    globals()[dag_id] = create_dag(commodity)
