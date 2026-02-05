"""
과거 뉴스 사례 검색 및 가격 변동 매칭 실행 모듈
"""

import os
import json
from datetime import date
from typing import List, Optional, Any, Dict
from google.cloud import bigquery

from app.models.pastnews_rag.price_jump import get_bq_client as _get_bq_client
from app.models.pastnews_rag.price_jump import fetch_article_dates, fetch_prices_for_dates


def extract_triples_from_today():
    """기본 검색용 triples"""
    return [["USDA", "announced", "export restrictions"]]


def fetch_embedding_by_triple_text(client, triple_text: str) -> Optional[List[float]]:
    """BigQuery에서 triple_text에 해당하는 임베딩 조회"""
    if not triple_text or not client:
        return None
    dataset = os.getenv("BIGQUERY_DATASET_ID", "tilda")
    table = "news_article_triples"
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
    """BigQuery VECTOR_SEARCH를 이용한 유사 기사 검색"""
    if not triple_embedding:
        return []
    dataset = os.getenv("BIGQUERY_DATASET_ID", "tilda")
    table = "news_article_triples"
    full_table = f"{client.project}.{dataset}.{table}"

    query = f"""
    SELECT base.hash_id
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
    **kwargs
) -> Dict[str, Any]:
    """
    현재 뉴스 키워드(triples)와 유사한 과거 사례를 찾고 당시의 가격 변동을 반환합니다.
    """
    result = {"article_info": []}

    # 입력 처리 및 팀원 추가 로직(앞 5개 제한) 적용
    if triples is None or len(triples) == 0:
        triples = extract_triples_from_today()
    
    triples = triples[:5] # 리소스 제한을 위한 상위 5개 사용

    # 첫 번째 triple 기반으로 벡터 검색 수행
    first_triple_list = triples[0]
    if not isinstance(first_triple_list, (list, tuple)) or len(first_triple_list) < 3:
        result["error"] = "Invalid triple format"
        return result
    
    triple_text = str(first_triple_list).strip()

    client = _get_bq_client()
    embedding = fetch_embedding_by_triple_text(client, triple_text)
    if not embedding:
        result["error"] = f"Embedding not found for {triple_text}"
        return result

    # 유사 기사 검색
    hash_ids = vector_search_similar_hash_ids(client, embedding, top_k=top_k)
    if not hash_ids:
        return result

    # 기사 정보 및 날짜 조회
    articles = fetch_article_dates(client, hash_ids)
    
    dates = []
    for r in articles:
        if hasattr(r, 'publish_date') and r.publish_date:
            dates.append(r.publish_date)
    
    # 품목별 가격 변동 매칭
    prices_by_date = {}
    if dates:
        prices = fetch_prices_for_dates(client, dates, commodity=commodity)
        for p in prices:
            d_str = str(p.base_date)
            if d_str not in prices_by_date: prices_by_date[d_str] = {}
            prices_by_date[d_str][p.offset_days] = float(p.close) if p.close else None
    
    # 최종 결과 구성
    for r in articles:
        d_str = str(r.publish_date)
        item = {
            "description": r.description,
            "publish_date": d_str,
            "0": prices_by_date.get(d_str, {}).get(0),
            "1": prices_by_date.get(d_str, {}).get(1),
            "3": prices_by_date.get(d_str, {}).get(3),
        }
        result["article_info"].append(item)

    return result