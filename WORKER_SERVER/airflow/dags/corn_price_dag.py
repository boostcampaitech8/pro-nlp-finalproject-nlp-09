from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta

import os
import sys

# 프로젝트 루트 경로를 path에 추가
sys.path.append('/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER')

from processor.bigquery.corn_price_uploader import upload_corn_price_for_date

default_args = {
    'owner': 'sehun',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 27),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'corn_price_daily_to_bq',
    default_args=default_args,
    description='TradingView 옥수수 가격 일봉 수집 후 BigQuery 적재',
    schedule='10 16 * * *',
    catchup=True,
    max_active_runs=1,
) as dag:

    def upload_task_func(**context):
        target_date = context["ds"]
        ema_period = int(os.getenv("CORN_EMA_PERIOD", "20"))
        symbol = os.getenv("TV_SYMBOL", "ZC1!")
        exchange = os.getenv("TV_EXCHANGE", "CBOT")

        dataset_id = os.getenv("BIGQUERY_DATASET_ID", "tilda")
        table_id = os.getenv("CORN_TABLE_ID", "corn_price")

        return upload_corn_price_for_date(
            target_date=target_date,
            ema_period=ema_period,
            symbol=symbol,
            exchange=exchange,
            dataset_id=dataset_id,
            table_id=table_id,
        )

    upload_corn_price = PythonOperator(
        task_id='upload_corn_price',
        python_callable=upload_task_func,
    )
