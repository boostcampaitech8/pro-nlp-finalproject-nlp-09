"""
step3_price_jump 기반: hash_id → article publish_date → 해당 일자 전후 가격 조회
"""

import os
from google.cloud import bigquery


def get_bq_client():
    project_id = os.getenv("BIGQUERY_PROJECT_ID") or os.getenv("VERTEX_AI_PROJECT_ID")
    if project_id:
        return bigquery.Client(project=project_id)
    return bigquery.Client()


def fetch_article_dates(client, hash_ids):
    """
    hash_ids로 triple_article_map에서 article_id를 가져온 후,
    news_article 테이블에서 description과 publish_date를 조회하여 반환합니다.
    
    Args:
        client: BigQuery 클라이언트
        hash_ids: triple hash_id 리스트
        
    Returns:
        list: 각 행은 description, publish_date 필드를 가진 Row 객체
    """
    if not hash_ids:
        return []
    dataset = os.getenv("BIGQUERY_DATASET_ID", "tilda")
    map_table = os.getenv("TRIPLE_MAP_TABLE", "triple_article_map")
    news_table = os.getenv("NEWS_TABLE", "news_article")
    map_full_table = f"{client.project}.{dataset}.{map_table}"
    news_full_table = f"{client.project}.{dataset}.{news_table}"
    
    query = f"""
    SELECT DISTINCT
      n.description,
      CAST(m.publish_date AS DATE) AS publish_date
    FROM `{map_full_table}` m
    JOIN `{news_full_table}` n
      ON m.article_id = n.id
    WHERE m.hash_id IN UNNEST(@hash_ids)
      AND m.publish_date IS NOT NULL
    """
    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("hash_ids", "STRING", hash_ids)]
        ),
    )
    return list(job.result())


def fetch_prices_for_dates(client, dates):
    """
    base_date(뉴스 날짜) 기준 당일(0), 1거래일후(1), 3거래일후(3) 종가 조회.
    time이 DATE든 TIMESTAMP든 항상 DATE( time )로 날짜만 쓰고, 날짜당 1행만 보장.
    """
    if not dates:
        return []
    dataset = os.getenv("BIGQUERY_DATASET_ID", "tilda")
    table = os.getenv("CORN_TABLE_ID", "corn_price")
    full_table = f"{client.project}.{dataset}.{table}"

    query = f"""
    WITH
    one_per_date AS (
      SELECT DATE(time) AS parsed_time, ANY_VALUE(close) AS close
      FROM `{full_table}`
      WHERE time IS NOT NULL
      GROUP BY DATE(time)
    ),
    base_dates AS (
      SELECT date AS base_date FROM UNNEST(@dates) AS date
    ),
    trading_days AS (
      SELECT
        b.base_date,
        p.parsed_time AS traded_date,
        ROW_NUMBER() OVER (PARTITION BY b.base_date ORDER BY p.parsed_time) - 1 AS trade_offset
      FROM base_dates b
      JOIN one_per_date p ON p.parsed_time >= b.base_date
    ),
    offsets AS (
      SELECT 0 AS offset_days UNION ALL SELECT 1 UNION ALL SELECT 3
    )
    SELECT
      t.base_date,
      o.offset_days,
      t.traded_date,
      p.close
    FROM offsets o
    JOIN trading_days t ON t.trade_offset = o.offset_days
    JOIN one_per_date p ON p.parsed_time = t.traded_date
    ORDER BY t.base_date, o.offset_days
    """
    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("dates", "DATE", dates)]
        ),
    )
    return list(job.result())
