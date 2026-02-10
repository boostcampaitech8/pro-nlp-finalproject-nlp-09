from airflow import DAG
from airflow.models import Variable
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
from hashlib import md5

import json
import os
import sys

# 프로젝트 루트 경로를 path에 추가 (crawler, processor 모듈을 불러오기 위함)
sys.path.append('/data/ephemeral/home/TeamProject-Repo/worker_server') 

from crawler.main_crawler import fetch_and_standardize
from processor.news_processor import NewsProcessor
from processor.embedder import TitanEmbedder
# 환경 설정 (Airflow Variables 우선, 없으면 환경변수)
OPENAI_API_KEY = Variable.get("OPENAI_API_KEY", default_var=None) or os.getenv("OPENAI_API_KEY")
DATA_DIR = "/data/ephemeral/home/TeamProject-Repo/worker_server/data" # 로컬 볼륨과 연결된 경로

os.makedirs(os.path.join(DATA_DIR, 'raw'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'processed'), exist_ok=True)

default_args = {
    'owner': 'sehun',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1), # 시작일 설정
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'agri_news_pipeline_v1',
    default_args=default_args,
    description='농산물 뉴스 수집 및 LLM 처리 파이프라인',
    schedule='@hourly', # 매시간 실행
    catchup=False
) as dag:

    def crawl_task_func(**context):
        # execution_date를 활용해 시간별 파일명 생성
        execution_date = context["ds_nodash"]
        hour = context["logical_date"].hour
        output_path = os.path.join(DATA_DIR, 'raw', f'news_{execution_date}_{hour}.json')
        
        # 크롤러 실행
        results = fetch_and_standardize()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        # 다음 태스크를 위해 경로 전달 (XCom)
        return output_path

    def process_task_func(**context):
        # 이전 태스크에서 생성된 파일 경로를 가져옴
        input_path = context['ti'].xcom_pull(task_ids='crawl_news')
        output_path = os.path.join(DATA_DIR, 'processed', 'final_processed_news.json')
        
        # 프로세서 실행
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set")
        processor = NewsProcessor(api_key=OPENAI_API_KEY)
        processor.process_json_file(input_path=input_path, output_path=output_path)

    def embed_news_task_func(**context):
        """작성하신 TitanEmbedder를 사용하여 null인 임베딩을 채우는 함수"""
        final_json_path = os.path.join(DATA_DIR, 'processed', 'final_processed_news.json')
        entity_json_path = os.path.join(DATA_DIR, 'processed', 'entity.json')
        triple_json_path = os.path.join(DATA_DIR, 'processed', 'triple.json')
        
        if not os.path.exists(final_json_path):
            print("❌ 처리할 JSON 파일이 없습니다.")
            return []

        # 1. 파일 읽기
        with open(final_json_path, 'r', encoding='utf-8') as f:
            all_articles = json.load(f)

        # 2. 임베더 초기화 (작성하신 클래스 사용)
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

        # 3. 임베딩이 없는 기사만 골라서 생성
        for art in all_articles:
            # T인 기사 중 임베딩이 null인 경우만!
            if art.get('article_embedding') is None:
                text_to_embed = f"{art['title']}\n\n{art.get('description', '')}"
                
                print(f"🔄 임베딩 생성 중: {art['title'][:20]}...")
                vector = embedder.generate_embedding(text_to_embed)
                
                if vector:
                    art['article_embedding'] = vector
                    newly_embedded_articles.append(art)

        # 4. 엔티티/트리플 임베딩 생성
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
                vector = embedder.generate_embedding(entity_text)
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
                vector = embedder.generate_embedding(triple_text)
                if vector:
                    new_triples.append({
                        "hash_id": triple_id,
                        "triple_text": triple_text,
                        "embedding": vector,
                    })
                    existing_triple_ids.add(triple_id)

        # 5. 파일 업데이트 저장
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

    # 1. 크롤링 태스크
    crawl_news = PythonOperator(
        task_id='crawl_news',
        python_callable=crawl_task_func,
    )

    # 2. LLM 처리 태스크
    process_news = PythonOperator(
        task_id='process_news',
        python_callable=process_task_func,
    )
    # 3. 임베딩 생성 태스크
    embed_news = PythonOperator(
        task_id='embed_news',
        python_callable=embed_news_task_func,
    )

    # 태스크 순서 설정
    crawl_news >> process_news >> embed_news
