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
    commodity: str = "corn",
    top_k: int = 5,
    dimensions: int = 1024,
) -> Dict[str, Any]:
    """
    top_triples 중 첫 번째 triple만 사용: tilda.news_article_triples에서 triple_text로
    embedding 조회 후 유사 hash_id 검색 → 해당 뉴스 description 조회.

    Args:
        triples: [[s, v, o], ...]. None이면 extract_triples_from_today() 사용.
        commodity: 상품명 (corn, soybean, wheat)
        top_k: 유사 hash_id 개수
        dimensions: (미사용, 호환용)

    Returns:
        dict: article_info (가격 데이터 포함)
    """
    result = {"article_info": []}

    if triples is None or len(triples) == 0:
        triples = extract_triples_from_today()
    if not triples:
        result["error"] = "triples가 비어 있습니다."
        return result

    # 첫 번째 triple만 사용
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

    if not hash_ids:
        return result

    articles = fetch_article_dates(client, hash_ids)
    
    # publish_date 추출 (DATE 타입으로 변환)
    dates = []
    for r in articles:
        if hasattr(r, 'publish_date') and r.publish_date:
            if isinstance(r.publish_date, date):
                dates.append(r.publish_date)
            elif isinstance(r.publish_date, str):
                try:
                    dates.append(date.fromisoformat(r.publish_date))
                except (ValueError, AttributeError):
                    pass
    
    # 가격 데이터 조회 (commodity 전달)
    prices_by_date = {}
    if dates:
        prices = fetch_prices_for_dates(client, dates, commodity=commodity)
        for price_row in prices:
            base_date_str = str(price_row.base_date)
            if base_date_str not in prices_by_date:
                prices_by_date[base_date_str] = {}
            offset = price_row.offset_days
            prices_by_date[base_date_str][offset] = float(price_row.close) if price_row.close is not None else None
    
    # article_info 구성 (가격 데이터 포함)
    result["article_info"] = []
    for r in articles:
        if not (hasattr(r, 'description') and r.description):
            continue
        
        publish_date_str = str(r.publish_date) if hasattr(r, 'publish_date') and r.publish_date else None
        if not publish_date_str:
            continue
        
        article_item = {
            "description": r.description,
            "publish_date": publish_date_str,
        }
        
        # 해당 날짜의 가격 데이터 추가
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
