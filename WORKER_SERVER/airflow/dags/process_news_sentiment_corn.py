from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google.cloud import bigquery

# [환경 설정]
PROJECT_ID = os.getenv("VERTEX_AI_PROJECT_ID", "project-5b75bb04-485d-454e-af7")
DATASET_ID = "tilda"
SOURCE_TABLE = "news_article"
TARGET_TABLE = "corn_all_news_with_sentiment"
COMMODITY = "corn"
MODEL_NAME = "ProsusAI/finbert"
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

def run_sentiment_analysis(**context):
    """Airflow 실행 날짜(ds)의 모든 뉴스를 분석합니다."""
    target_date_str = context['ds']
    logging.info(f"Target Date: {target_date_str} 뉴스 분석 시작")
    
    client = get_bq_client()
    
    # 1. 해당 날짜의 원본 뉴스 조회 (filter_status='T' AND corn 관련)
    q_source = f"""
        SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{SOURCE_TABLE}`
        WHERE filter_status = 'T'
          AND publish_date = '{target_date_str}'
          AND (LOWER(key_word) LIKE '%corn%')
          AND (LOWER(key_word) LIKE '%price%' OR LOWER(key_word) LIKE '%demand%' OR LOWER(key_word) LIKE '%supply%' OR LOWER(key_word) LIKE '%inventory%')
    """
    df = client.query(q_source).to_dataframe()
    
    if df.empty:
        logging.info(f"{target_date_str}에 분석할 뉴스가 없습니다.")
        return None

    # 2. 모델 로딩
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
    results = []

    # 3. 뉴스별 감성 분석 수행
    for _, row in df.iterrows():
        text = f"{row['title']} {row.get('description', '')}".strip()
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        
        pred_class = np.argmax(probs)
        
        res_row = row.to_dict()
        res_row['combined_text'] = text
        res_row['sentiment'] = label_map[pred_class]
        res_row['sentiment_confidence'] = float(probs[pred_class])
        res_row['positive_score'] = float(probs[0])
        res_row['negative_score'] = float(probs[1])
        res_row['neutral_score'] = float(probs[2])
        res_row['price_impact_score'] = float(probs[0] - probs[1])
        
        # 날짜 포맷 정리
        if isinstance(res_row['publish_date'], (datetime, pd.Timestamp)):
            res_row['publish_date'] = res_row['publish_date'].strftime('%Y-%m-%d')
        
        results.append(res_row)

    return results

def insert_to_bq(**context):
    """결과 적재 (동일 날짜 재실행 시 기존 데이터 삭제)"""
    results = context['ti'].xcom_pull(task_ids='run_sentiment_analysis')
    if not results: return

    client = get_bq_client()
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TARGET_TABLE}"
    target_date = context['ds']
    
    # 중복 방지: 해당 날짜 데이터 삭제
    try:
        client.query(f"DELETE FROM `{table_id}` WHERE publish_date = '{target_date}'").result()
    except Exception: pass

    # 적재
    clean_results = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in results]
    errors = client.insert_rows_json(table_id, clean_results)
    if errors: raise RuntimeError(f"BQ 적재 실패: {errors}")
    logging.info(f"{target_date} 데이터 적재 완료 ({len(clean_results)}건)")

with DAG(
    'process_news_sentiment_corn_v1',
    default_args=default_args,
    description='Corn 뉴스 감성 분석 (Backfill 지원)',
    schedule_interval='0 17 * * *', # 매일 17:00 (시계열 피처 적재 후 실행 권장)
    catchup=True,
    max_active_runs=1,
    tags=['news', 'sentiment', 'corn', 'finbert']
) as dag:

    t1 = PythonOperator(task_id='run_sentiment_analysis', python_callable=run_sentiment_analysis, provide_context=True)
    t2 = PythonOperator(task_id='insert_to_bq', python_callable=insert_to_bq, provide_context=True)

    t1 >> t2