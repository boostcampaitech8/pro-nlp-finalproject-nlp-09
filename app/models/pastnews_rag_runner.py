"""
triple_text로 tilda.news_article_triples에서 embedding 조회 → 유사 hash_id 검색 → 뉴스 일자 전후 가격 조회
"""

import os
from typing import List, Optional, Any, Dict

from google.cloud import bigquery

from app.models.pastnews_rag import get_bq_client, fetch_article_dates, fetch_prices_for_dates


def _get_bq_client():
    return get_bq_client()


def extract_triples_from_today():
    """triples 없을 때 사용할 예시 1개 반환 (BQ triple_text 조회용)"""
    return [["USDA", "announced", "corn export restrictions"]]


def fetch_embedding_by_triple_text(client, triple_text: str) -> Optional[List[float]]:
    """BigQuery tilda.news_article_triples에서 triple_text로 행을 찾아 embedding 반환."""
    if not triple_text or not client:
        return None
    dataset = os.getenv("BIGQUERY_DATASET_ID", "tilda")
    table = os.getenv("TRIPLES_TABLE", "news_article_triples")
    full_table = f"{client.project}.{dataset}.{table}"
    query = f"""
    SELECT embedding
    FROM `{full_table}`
    WHERE triple_text = @triple_text
    LIMIT 1
    """
    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("triple_text", "STRING", triple_text)
            ]
        ),
    )
    rows = list(job.result())
    if not rows or not rows[0].embedding:
        return None
    return list(rows[0].embedding)


def vector_search_similar_hash_ids(client, triple_embedding: List[float], top_k: int = 5) -> List[str]:
    """BigQuery VECTOR_SEARCH로 유사 hash_id 검색"""
    if not triple_embedding:
        return []
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


def run_pastnews_rag(
    triples: Optional[List[List[str]]] = None,
    top_k: int = 5,
    dimensions: int = 1024,
) -> Dict[str, Any]:
    """
    top_triples 중 첫 번째 triple만 사용: tilda.news_article_triples에서 triple_text로
    embedding 조회 후 유사 hash_id 검색 → 해당 뉴스 publish_date 전후 가격 조회.

    Args:
        triples: [[s, v, o], ...]. None이면 extract_triples_from_today() 사용.
        top_k: 유사 hash_id 개수
        dimensions: (미사용, 호환용)

    Returns:
        dict: hash_ids, article_mappings, price_data, error(있을 경우)
    """
    result = {"hash_ids": [], "article_mappings": [], "price_data": []}

    if triples is None or len(triples) == 0:
        triples = extract_triples_from_today()
    if not triples:
        result["error"] = "triples가 비어 있습니다."
        return result

    # 첫 번째 triple만 사용 (DAG 저장 형식과 동일하게 str(triple).strip())
    first_triple_list = triples[0]
    if not isinstance(first_triple_list, (list, tuple)) or len(first_triple_list) < 3:
        result["error"] = "첫 triple이 [s, v, o] 형식이 아닙니다."
        return result
    triple_text = str(first_triple_list).strip()

    client = _get_bq_client()
    embedding = fetch_embedding_by_triple_text(client, triple_text)
    if not embedding:
        result["error"] = (
            f"tilda.news_article_triples에서 triple_text로 embedding을 찾을 수 없습니다. (triple_text={triple_text!r})"
        )
        return result

    hash_ids = vector_search_similar_hash_ids(client, embedding, top_k=top_k)
    result["hash_ids"] = hash_ids

    if not hash_ids:
        return result

    mappings = fetch_article_dates(client, hash_ids)
    result["article_mappings"] = [
        {"hash_id": r.hash_id, "article_id": r.article_id, "publish_date": str(r.publish_date)}
        for r in mappings
    ]

    dates = [r.publish_date for r in mappings]
    if not dates:
        return result

    prices = fetch_prices_for_dates(client, dates)
    result["price_data"] = [
        {
            "base_date": str(r.base_date),
            "offset_days": r.offset_days,
            "traded_date": str(r.traded_date),
            "close": float(r.close) if r.close is not None else None,
        }
        for r in prices
    ]

    return result
