"""
triple_text로 tilda.news_article_triples에서 embedding 조회 → 유사 hash_id 검색 → 뉴스 일자 전후 가격 조회
"""

import os
from datetime import date
from typing import List, Optional, Any, Dict

from google.cloud import bigquery

from app.models.pastnews_rag.price_jump import get_bq_client as _get_bq_client
from app.models.pastnews_rag.price_jump import fetch_article_dates, fetch_prices_for_dates



def extract_triples_from_today():
    """triples 없을 때 사용할 예시 1개 반환 (BQ triple_text 조회용)"""
    return [["USDA", "announced", "corn export restrictions"]]


def fetch_embedding_by_triple_text(client, triple_text: str) -> Optional[List[float]]:
    """BigQuery tilda.news_article_triples에서 triple_text로 행을 찾아 embedding 반환."""
    if not triple_text or not client:
        return None
    result = fetch_embeddings_by_triple_texts(client, [triple_text])
    return result.get(triple_text)


def fetch_embeddings_by_triple_texts(
    client, triple_texts: List[str]
) -> Dict[str, List[float]]:
    """
    BigQuery tilda.news_article_triples에서 triple_text 목록으로 행을 찾아
    triple_text -> embedding 매핑을 반환. (동일 triple_text 복수 행 시 첫 행만 사용)
    """
    out: Dict[str, List[float]] = {}
    if not client or not triple_texts:
        return out
    unique_texts = list(dict.fromkeys(t for t in triple_texts if t))
    if not unique_texts:
        return out
    dataset = os.getenv("BIGQUERY_DATASET_ID", "tilda")
    table = os.getenv("TRIPLES_TABLE", "news_article_triples")
    full_table = f"{client.project}.{dataset}.{table}"
    query = f"""
    SELECT triple_text, embedding
    FROM `{full_table}`
    WHERE triple_text IN UNNEST(@triple_texts)
    """
    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("triple_texts", "STRING", unique_texts)
            ]
        ),
    )
    for row in job.result():
        if row.triple_text not in out and row.embedding:
            out[row.triple_text] = list(row.embedding)
    return out


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
    top_k: int = 2,
    dimensions: int = 1024,
) -> Dict[str, Any]:
    """
    모든 triple을 사용: 각 triple마다 embedding 조회 후 유사 hash_id 검색(triple당 최대 top_k개)
    → 과거 뉴스 수집 후 최신순 정렬. 연관 키워드는 keyword_analyzer의 top_triples를 보고서에서 사용하면 됨.

    Args:
        triples: [[s, v, o], ...]. None이면 extract_triples_from_today() 사용.
        top_k: triple당 가져올 유사 hash_id 개수 (기본 2)
        dimensions: (미사용, 호환용)

    Returns:
        dict: article_info (각 항목: description, publish_date, 0, 1, 3), error(있을 경우)
    """
    result = {"article_info": []}
    max_per_triple = max(1, min(top_k, 10))

    if triples is None or len(triples) == 0:
        triples = extract_triples_from_today()
    if not triples:
        result["error"] = "triples가 비어 있습니다."
        return result

    triple_texts = []
    for t in triples:
        if not isinstance(t, (list, tuple)) or len(t) < 3:
            continue
        triple_texts.append(str(t).strip())
    if not triple_texts:
        result["error"] = "유효한 [s, v, o] 형식의 triple이 없습니다."
        return result

    client = _get_bq_client()
    embeddings_map = fetch_embeddings_by_triple_texts(client, triple_texts)
    if not embeddings_map:
        result["error"] = (
            "tilda.news_article_triples에서 triple_text로 embedding을 찾을 수 없습니다."
        )
        return result

    all_hash_ids = []
    for tt in triple_texts:
        emb = embeddings_map.get(tt)
        if not emb:
            continue
        ids = vector_search_similar_hash_ids(client, emb, top_k=max_per_triple)
        all_hash_ids.extend(ids)
    hash_ids = list(dict.fromkeys(all_hash_ids))
    if not hash_ids:
        return result

    articles = fetch_article_dates(client, hash_ids)

    def _sort_key(r):
        pd = getattr(r, "publish_date", None)
        if pd is None:
            return (0, None)
        d = pd if isinstance(pd, date) else (date.fromisoformat(pd) if isinstance(pd, str) else None)
        return (1, d) if d else (0, None)

    articles = sorted(articles, key=_sort_key, reverse=True)

    dates = []
    for r in articles:
        if hasattr(r, "publish_date") and r.publish_date:
            if isinstance(r.publish_date, date):
                dates.append(r.publish_date)
            elif isinstance(r.publish_date, str):
                try:
                    dates.append(date.fromisoformat(r.publish_date))
                except (ValueError, AttributeError):
                    pass

    prices_by_date = {}
    if dates:
        prices = fetch_prices_for_dates(client, dates)
        for price_row in prices:
            base_date_str = str(price_row.base_date)
            if base_date_str not in prices_by_date:
                prices_by_date[base_date_str] = {}
            offset = price_row.offset_days
            prices_by_date[base_date_str][offset] = float(price_row.close) if price_row.close is not None else None

    result["article_info"] = []
    for r in articles:
        if not (hasattr(r, "description") and r.description):
            continue
        publish_date_str = str(r.publish_date) if hasattr(r, "publish_date") and r.publish_date else None
        if not publish_date_str:
            continue
        article_item = {
            "description": r.description,
            "publish_date": publish_date_str,
        }
        if publish_date_str in prices_by_date:
            price_data = prices_by_date[publish_date_str]
            article_item["0"] = price_data.get(0)
            article_item["1"] = price_data.get(1)
            article_item["3"] = price_data.get(3)
        else:
            article_item["0"] = None
            article_item["1"] = None
            article_item["3"] = None
        result["article_info"].append(article_item)

    return result
