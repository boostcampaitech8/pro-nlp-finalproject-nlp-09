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
    top_k: int = 2,
    dimensions: int = 1024,
    target_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    모든 triple을 사용: 각 triple마다 embedding 조회 후 유사 hash_id 검색(triple당 최대 top_k개)
    → 과거 뉴스 수집 후 최신순 정렬. target_date 기준 최소 3일 이전 기사만 포함하며,
    앞 100자가 같은 기사는 중복으로 간주하고 가장 최근 1건만 유지합니다.

    Args:
        triples: [[s, v, o], ...]. None이면 extract_triples_from_today() 사용.
        commodity: 상품명 (corn, soybean, wheat)
        top_k: triple당 가져올 유사 hash_id 개수 (기본 2)
        dimensions: (미사용, 호환용)
        target_date: 기준일 (YYYY-MM-DD). None이면 오늘. 이 날짜 기준 최소 3일 이전 기사만 포함.

    Returns:
        dict: article_info (가격 데이터 포함)
    """
    result = {"article_info": []}
    # triple 개수: 호출부(triples 인자)에서 결정. triple당 기사 수: 에이전트 top_k와 무관하게 항상 2개로 제한.
    max_per_triple = 2

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
    triple_texts = triple_texts[:7]
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

    # target_date가 있으면 BigQuery 조회 시점에 최소 3일 이전 기사만 가져옴
    articles = fetch_article_dates(client, hash_ids, target_date=target_date)

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
        prices = fetch_prices_for_dates(client, dates, commodity=commodity)
        for price_row in prices:
            base_date_str = str(price_row.base_date)
            if base_date_str not in prices_by_date:
                prices_by_date[base_date_str] = {}
            offset = price_row.offset_days
            prices_by_date[base_date_str][offset] = float(price_row.close) if price_row.close is not None else None

    def _normalize_prefix(s: str, length: int = 50) -> str:
        """중복 판별용: 공백 정규화 후 앞 length자. 다른 날짜 같은 기사 제거용."""
        if not s:
            return ""
        normalized = " ".join(str(s).split())
        return normalized[:length]

    result["article_info"] = []
    max_articles = 10
    seen_prefix: set = set()  # 앞 50자(정규화) 기준 중복 제거, 최신순이라 첫 등장이 가장 최근
    for r in articles:
        if len(result["article_info"]) >= max_articles:
            break
        # all_text 우선, 없으면 description 사용
        text_val = None
        if hasattr(r, "all_text") and r.all_text:
            text_val = r.all_text
        elif hasattr(r, "description") and r.description:
            text_val = r.description
        if not text_val:
            continue
        prefix = _normalize_prefix(text_val, 50)
        if prefix in seen_prefix:
            continue
        seen_prefix.add(prefix)
        publish_date_str = str(r.publish_date) if hasattr(r, "publish_date") and r.publish_date else None
        if not publish_date_str:
            continue
        article_item = {
            "all_text": text_val,
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