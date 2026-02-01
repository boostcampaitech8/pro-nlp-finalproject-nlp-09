import os
from datetime import datetime, timedelta

from google.cloud import bigquery

try:
    from processor.embedder import TitanEmbedder
except ImportError:
    from embedder import TitanEmbedder


def _get_bq_client():
    project_id = os.getenv("BIGQUERY_PROJECT_ID") or os.getenv("VERTEX_AI_PROJECT_ID")
    if project_id:
        return bigquery.Client(project=project_id)
    return bigquery.Client()


def extract_triples_from_today():
    # Step 1: LLM 추출 자리 (예시 1개만 반환)
    return [["Brazil", "drought persists", "coffee production declines"]]


def embed_triples(triples, dimensions=1024):
    # Step 1-2: 임베딩 생성 (TitanEmbedder 사용)
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
    SELECT hash_id, distance
    FROM VECTOR_SEARCH(
      TABLE `{full_table}`,
      'embedding',
      @embedding,
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
    results = list(job.result())
    return [row.hash_id for row in results]


def fetch_dates_for_hash_ids(client, hash_ids):
    if not hash_ids:
        return []
    dataset = os.getenv("BIGQUERY_DATASET_ID", "tilda")
    table = os.getenv("TRIPLE_MAP_TABLE", "triple_article_map")
    full_table = f"{client.project}.{dataset}.{table}"
    query = f"""
    SELECT DISTINCT publish_date
    FROM `{full_table}`
    WHERE hash_id IN UNNEST(@hash_ids)
    """
    job = client.query(query, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("hash_ids", "STRING", hash_ids)]
    ))
    return [row.publish_date for row in job.result()]


def fetch_prices_for_dates(client, dates):
    if not dates:
        return []
    dataset = os.getenv("BIGQUERY_DATASET_ID", "tilda")
    table = os.getenv("CORN_TABLE_ID", "corn_price")
    full_table = f"{client.project}.{dataset}.{table}"

    # D+0, D+1, D+3 가격 조회
    query = f"""
    WITH base_dates AS (
      SELECT date as base_date
      FROM UNNEST(@dates) as date
    ),
    offsets AS (
      SELECT base_date, base_date as target_date FROM base_dates
      UNION ALL SELECT base_date, DATE_ADD(base_date, INTERVAL 1 DAY) FROM base_dates
      UNION ALL SELECT base_date, DATE_ADD(base_date, INTERVAL 3 DAY) FROM base_dates
    )
    SELECT o.base_date, o.target_date, p.close
    FROM offsets o
    LEFT JOIN `{full_table}` p
    ON p.time = o.target_date
    ORDER BY o.base_date, o.target_date
    """
    job = client.query(query, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("dates", "DATE", dates)]
    ))
    return list(job.result())


def compute_risk(prices):
    # Step 4: 간단 예시 로직
    # 실제로는 D+1 변화율 분포로 리스크 계산
    changes = []
    grouped = {}
    for row in prices:
        grouped.setdefault(row.base_date, {})[row.target_date] = row.close
    for base_date, data in grouped.items():
        d0 = data.get(base_date)
        d1 = data.get(base_date + timedelta(days=1))
        if d0 and d1:
            changes.append((d1 - d0) / d0)
    if not changes:
        return {"status": "no_data"}
    avg = sum(changes) / len(changes)
    variance = sum((c - avg) ** 2 for c in changes) / len(changes)
    return {"avg_change": avg, "volatility": variance ** 0.5, "samples": len(changes)}


def run_example():
    # Step 1
    triples = extract_triples_from_today()
    # Step 2 (embed + vector search)
    embeddings = embed_triples(triples)
    hash_ids = vector_search_similar_hash_ids(embeddings[tuple(triples[0])], top_k=5)

    print("Hash IDs:", hash_ids)

    # Step 3 + 4 (BigQuery 필요)
    client = _get_bq_client()
    dates = fetch_dates_for_hash_ids(client, hash_ids)
    prices = fetch_prices_for_dates(client, dates)
    risk = compute_risk(prices)
    print("Risk:", risk)


if __name__ == "__main__":
    run_example()
