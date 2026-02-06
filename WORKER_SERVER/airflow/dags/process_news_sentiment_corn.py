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
PROJECT_ID = os.getenv("VERTEX_AI_PROJECT_ID", "team-blue-448407")
DATASET_ID = "tilda"
SOURCE_TABLE = "news_article"
TARGET_TABLE = "corn_all_news_with_sentiment"
COMMODITY = "corn"
MODEL_NAME = "ProsusAI/finbert"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def get_bq_client():
    return bigquery.Client(project=PROJECT_ID)

def check_new_news(**context):
    """분석이 필요한 신규 뉴스가 있는지 확인"""
    client = get_bq_client()
    
    # 1. 이미 분석된 뉴스 ID 목록 가져오기 (최근 30일치만 확인하여 속도 최적화)
    q_existing = f"SELECT id FROM `{PROJECT_ID}.{DATASET_ID}.{TARGET_TABLE}` WHERE publish_date >= CURRENT_DATE() - 30"
    try:
        existing_ids = set(client.query(q_existing).to_dataframe()['id'].tolist())
    except Exception:
        existing_ids = set()

    # 2. 원본 테이블에서 분석 대상 뉴스 조회
    # 조건: filter_status='T' AND corn 관련 키워드
    q_source = f"""
        SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{SOURCE_TABLE}`
        WHERE filter_status = 'T'
          AND (LOWER(key_word) LIKE '%corn%')
          AND (LOWER(key_word) LIKE '%price%' OR LOWER(key_word) LIKE '%demand%' OR LOWER(key_word) LIKE '%supply%' OR LOWER(key_word) LIKE '%inventory%')
          AND publish_date >= CURRENT_DATE() - 7
    """
    df_source = client.query(q_source).to_dataframe()
    
    if df_source.empty:
        logging.info("분석할 뉴스가 없습니다.")
        return False

    # 3. 중복 제외한 신규 뉴스 필터링
    df_new = df_source[~df_source['id'].isin(existing_ids)].copy()
    
    if df_new.empty:
        logging.info("모든 뉴스가 이미 분석되었습니다.")
        return False

    # 분석 대상 데이터를 JSON으로 변환하여 XCom 저장 (용량 주의: 100건 제한)
    df_new = df_new.head(100)
    context['ti'].xcom_push(key='news_data', value=df_new.to_json(orient='records'))
    logging.info(f"신규 뉴스 {len(df_new)}건 발견")
    return True

def run_sentiment_analysis(**context):
    """FinBERT 모델을 사용하여 감성 분석 수행"""
    news_json = context['ti'].xcom_pull(key='news_data', task_ids='check_new_news')
    if not news_json: return None
    
    df = pd.read_json(news_json)
    
    logging.info(f"모델 로딩 중: {MODEL_NAME}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
    results = []

    for _, row in df.iterrows():
        # 텍스트 결합 (title + description)
        text = f"{row['title']} {row.get('description', '')}".strip()
        
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        
        pred_class = np.argmax(probs)
        
        # 결과 매핑 (20개 컬럼 스키마 준수)
        res_row = row.to_dict()
        res_row['combined_text'] = text
        res_row['sentiment'] = label_map[pred_class]
        res_row['sentiment_confidence'] = float(probs[pred_class])
        res_row['positive_score'] = float(probs[0])
        res_row['negative_score'] = float(probs[1])
        res_row['neutral_score'] = float(probs[2])
        res_row['price_impact_score'] = float(probs[0] - probs[1])
        
        # BQ 적재를 위해 datetime 포맷 보정
        if isinstance(res_row['publish_date'], (int, np.integer)):
            res_row['publish_date'] = datetime.fromtimestamp(res_row['publish_date']/1000).strftime('%Y-%m-%d')
        
        results.append(res_row)

    logging.info("감성 분석 완료")
    return results

def insert_to_bq(**context):
    """분석 결과 적재"""
    results = context['ti'].xcom_pull(task_ids='run_sentiment_analysis')
    if not results: return

    client = get_bq_client()
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{TARGET_TABLE}"
    
    # 데이터 타입 클렌징 (NaN -> None)
    clean_results = []
    for r in results:
        clean_row = {k: (None if pd.isna(v) else v) for k, v in r.items()}
        clean_results.append(clean_row)

    errors = client.insert_rows_json(table_id, clean_results)
    if errors: raise RuntimeError(f"BQ 적재 실패: {errors}")
    logging.info(f"{len(clean_results)}건 적재 완료")

with DAG(
    'process_news_sentiment_corn_v1',
    default_args=default_args,
    description='Corn 뉴스 감성 분석 및 적재 (Incremental)',
    schedule_interval='15 * * * *',
    catchup=False,
    tags=['news', 'sentiment', 'corn', 'finbert']
) as dag:

    t1 = PythonOperator(task_id='check_new_news', python_callable=check_new_news, provide_context=True)
    t2 = PythonOperator(task_id='run_sentiment_analysis', python_callable=run_sentiment_analysis, provide_context=True)
    t3 = PythonOperator(task_id='insert_to_bq', python_callable=insert_to_bq, provide_context=True)

    t1 >> t2 >> t3
