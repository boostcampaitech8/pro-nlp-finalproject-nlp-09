import os
import sys

from google.cloud import bigquery

sys.path.append('/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER')

from processor.embedder import TitanEmbedder


def _get_bq_client():
    project_id = os.getenv("BIGQUERY_PROJECT_ID") or os.getenv("VERTEX_AI_PROJECT_ID")
    if project_id:
        return bigquery.Client(project=project_id)
    return bigquery.Client()


def extract_triples_from_today():
    # Step 1: LLM 추출 자리 (예시 1개만 반환)
    return [["USDA", "announced", "corn export restrictions"]]


def embed_triples(triples, dimensions=1024):
    # Step 2 준비: 임베딩 생성 (TitanEmbedder 사용)
    embedder = TitanEmbedder(region_name="us-east-1")
    embeddings = {}
    for triple in triples:
        text = " | ".join(triple)
        vector = embedder.generate_embedding(text, dimensions=dimensions)
        embeddings[tuple(triple)] = vector
    return embeddings


def vector_search_similar_hash_ids(triple_embedding, top_k=5):
    # Step 2: BigQuery VECTOR_SEARCH로 유사 hash_id 검색
    client = _get_bq_client()
    dataset = os.getenv("BIGQUERY_DATASET_ID", "tilda")
    table = os.getenv("TRIPLES_TABLE", "news_article_triples")
    full_table = f"{client.project}.{dataset}.{table}"

    query = f"""
    SELECT base.hash_id, distance
    FROM VECTOR_SEARCH(
      TABLE `{full_table}`,
      'embedding',
      (SELECT @embedding AS embedding),
      top_k => {top_k}
    )
    """
    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("embedding", "FLOAT64", triple_embedding)
            ]
        ),
    )
    return [row.hash_id for row in job.result()]


def run_step2_only():
    triples = extract_triples_from_today()
    embeddings = embed_triples(triples)
    hash_ids = vector_search_similar_hash_ids(embeddings[tuple(triples[0])], top_k=5)
    print("Hash IDs:", hash_ids)


if __name__ == "__main__":
    run_step2_only()
