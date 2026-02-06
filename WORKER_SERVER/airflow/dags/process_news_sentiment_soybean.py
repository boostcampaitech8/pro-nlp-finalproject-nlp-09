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
TARGET_TABLE = "soybean_all_news_with_sentiment"
COMMODITY = "soybean"
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
    target_date_str = context['ds']
    logging.info(f"Target Date: {target_date_str} 뉴스 분석 시작")
    client = get_bq_client()
    
    q_source = f"""
        SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{SOURCE_TABLE}`
        WHERE filter_status = 'T'
          AND publish_date = '{target_date_str}'
          AND (LOWER(key_word) LIKE '%soybean%')
          AND (LOWER(key_word) LIKE '%price%' OR LOWER(key_word) LIKE '%demand%' OR LOWER(key_word) LIKE '%supply%' OR LOWER(key_word) LIKE '%inventory%')
    """
    df = client.query(q_source).to_dataframe()
    if df.empty: return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
    results = []

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
        
        if isinstance(res_row['publish_date'], (datetime, pd.Timestamp)):
            res_row['publish_date'] = res_row['publish_date'].strftime('%Y-%m-%d')
        results.append(res_row)
    return results

def insert_to_bq(**context):
    results = context['ti'].xcom_pull(task_ids='run_sentiment_analysis')
    if not results: return
    client = get_bq_client()
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TARGET_TABLE}"
    target_date = context['ds']
    try:
        client.query(f"DELETE FROM `{table_id}` WHERE publish_date = '{target_date}'").result()
    except Exception: pass
    clean_results = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in results]
    errors = client.insert_rows_json(table_id, clean_results)
    if errors: raise RuntimeError(f"BQ 적재 실패: {errors}")

with DAG(
    'process_news_sentiment_soybean_v1',
    default_args=default_args,
    description='Soybean 뉴스 감성 분석 (Backfill 지원)',
    schedule_interval='10 17 * * *', # 매일 17:10
    catchup=True,
    max_active_runs=1,
    tags=['news', 'sentiment', 'soybean', 'finbert']
) as dag:
    t1 = PythonOperator(task_id='run_sentiment_analysis', python_callable=run_sentiment_analysis, provide_context=True)
    t2 = PythonOperator(task_id='insert_to_bq', python_callable=insert_to_bq, provide_context=True)
    t1 >> t2