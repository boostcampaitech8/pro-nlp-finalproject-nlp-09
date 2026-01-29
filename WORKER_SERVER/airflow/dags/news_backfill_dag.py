from airflow import DAG
from airflow.models import Variable
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
from hashlib import md5

import json
import os
import sys

# 프로젝트 루트 경로를 path에 추가 (crawler, processor 모듈을 불러오기 위함)
sys.path.append('/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER')

from crawler.backfill_crawler import fetch_and_standardize_backfill
from processor.news_processor import NewsProcessor
from processor.embedder import TitanEmbedder

# 환경 설정 (Airflow Variables 우선, 없으면 환경변수)
OPENAI_API_KEY = Variable.get("OPENAI_API_KEY", default_var=None) or os.getenv("OPENAI_API_KEY")
DATA_DIR = "/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER/data"

os.makedirs(os.path.join(DATA_DIR, 'raw'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'processed'), exist_ok=True)

default_args = {
    'owner': 'sehun',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 13),
    'end_date': datetime(2026, 1, 28),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'agri_news_backfill_v1',
    default_args=default_args,
    description='농산물 뉴스 백필 수집 파이프라인(일 단위)',
    schedule='@daily',
    catchup=True,
    max_active_runs=1,
) as dag:

    def crawl_task_func(**context):
        execution_date = context["logical_date"].date()
        execution_date_nodash = context["ds_nodash"]

        print(f"🚦 crawl_news_backfill start: {execution_date}")
        end_date = execution_date
        output_name = f'news_backfill_{execution_date_nodash}.json'

        output_path = os.path.join(DATA_DIR, 'raw', output_name)

        results = fetch_and_standardize_backfill(
            start_date=execution_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            output_path=output_path,
        )
        if not results:
            print("⚠️ 수집된 뉴스가 없습니다.")
        return output_path

    def process_task_func(**context):
        input_path = context['ti'].xcom_pull(task_ids='crawl_news_backfill')
        output_path = os.path.join(DATA_DIR, 'processed', f'processed_news_{context["ds_nodash"]}.json')

        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set")
        processor = NewsProcessor(api_key=OPENAI_API_KEY)
        processor.process_json_file(input_path=input_path, output_path=output_path)
        return output_path

    def embed_news_task_func(**context):
        final_json_path = context['ti'].xcom_pull(task_ids='process_news_backfill')
        entity_json_path = os.path.join(DATA_DIR, 'processed', 'entity.json')
        triple_json_path = os.path.join(DATA_DIR, 'processed', 'triple.json')

        if not os.path.exists(final_json_path):
            print("❌ 처리할 JSON 파일이 없습니다.")
            return []

        with open(final_json_path, 'r', encoding='utf-8') as f:
            all_articles = json.load(f)

        embedder = TitanEmbedder(region_name="us-east-1")
        newly_embedded_articles = []
        new_entities = []
        new_triples = []

        existing_entity_ids = set()
        existing_triple_ids = set()

        if os.path.exists(entity_json_path):
            with open(entity_json_path, 'r', encoding='utf-8') as f:
                try:
                    existing_entities = json.load(f)
                    existing_entity_ids = {e.get("hash_id") for e in existing_entities if e.get("hash_id")}
                except json.JSONDecodeError:
                    existing_entities = []
        else:
            existing_entities = []

        if os.path.exists(triple_json_path):
            with open(triple_json_path, 'r', encoding='utf-8') as f:
                try:
                    existing_triples = json.load(f)
                    existing_triple_ids = {t.get("hash_id") for t in existing_triples if t.get("hash_id")}
                except json.JSONDecodeError:
                    existing_triples = []
        else:
            existing_triples = []

        for art in all_articles:
            if art.get('article_embedding') is None:
                text_to_embed = f"{art['title']}\n\n{art.get('description', '')}"
                print(f"🔄 임베딩 생성 중: {art['title'][:20]}...")
                vector = embedder.generate_embedding(text_to_embed, dimensions=512)
                if vector:
                    art['article_embedding'] = vector
                    newly_embedded_articles.append(art)

        def compute_mdhash_id(content: str, prefix: str) -> str:
            return prefix + md5(content.encode()).hexdigest()

        for art in newly_embedded_articles:
            for entity in art.get("named_entities", []) or []:
                entity_text = str(entity).strip()
                if not entity_text:
                    continue
                entity_id = compute_mdhash_id(entity_text, prefix="entity-")
                if entity_id in existing_entity_ids:
                    continue
                vector = embedder.generate_embedding(entity_text, dimensions=1024)
                if vector:
                    new_entities.append({
                        "hash_id": entity_id,
                        "entity_text": entity_text,
                        "embedding": vector,
                    })
                    existing_entity_ids.add(entity_id)

            for triple in art.get("triples", []) or []:
                triple_text = str(triple).strip()
                if not triple_text:
                    continue
                triple_id = compute_mdhash_id(triple_text, prefix="triple-")
                if triple_id in existing_triple_ids:
                    continue
                vector = embedder.generate_embedding(triple_text, dimensions=1024)
                if vector:
                    new_triples.append({
                        "hash_id": triple_id,
                        "triple_text": triple_text,
                        "embedding": vector,
                    })
                    existing_triple_ids.add(triple_id)

        if newly_embedded_articles:
            with open(final_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_articles, f, indent=4, ensure_ascii=False)
            print(f"✅ {len(newly_embedded_articles)}개의 뉴스 임베딩 완료!")

        if new_entities:
            with open(entity_json_path, 'w', encoding='utf-8') as f:
                json.dump(existing_entities + new_entities, f, indent=4, ensure_ascii=False)
            print(f"✅ {len(new_entities)}개의 엔티티 임베딩 완료!")

        if new_triples:
            with open(triple_json_path, 'w', encoding='utf-8') as f:
                json.dump(existing_triples + new_triples, f, indent=4, ensure_ascii=False)
            print(f"✅ {len(new_triples)}개의 트리플 임베딩 완료!")

        return newly_embedded_articles

    crawl_news_backfill = PythonOperator(
        task_id='crawl_news_backfill',
        python_callable=crawl_task_func,
    )

    process_news_backfill = PythonOperator(
        task_id='process_news_backfill',
        python_callable=process_task_func,
    )

    embed_news_backfill = PythonOperator(
        task_id='embed_news_backfill',
        python_callable=embed_news_task_func,
    )

    crawl_news_backfill >> process_news_backfill >> embed_news_backfill
