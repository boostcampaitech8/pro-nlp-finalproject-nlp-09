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
    if not dates:
        return []
    dataset = os.getenv("BIGQUERY_DATASET_ID", "tilda")
    table = os.getenv("CORN_TABLE_ID", "corn_price")
    full_table = f"{client.project}.{dataset}.{table}"

    query = f"""
    WITH base_dates AS (
      SELECT date as base_date
      FROM UNNEST(@dates) as date
    ),
    price_dates AS (
      SELECT DISTINCT 
        CAST(time AS DATE) AS parsed_time,
        time AS original_time
      FROM `{full_table}`
    ),
    trading_days AS (
      SELECT
        b.base_date,
        pd.parsed_time AS traded_date,
        ROW_NUMBER() OVER (PARTITION BY b.base_date ORDER BY pd.parsed_time) - 1 AS trade_offset
      FROM base_dates b
      JOIN price_dates pd
        ON pd.parsed_time >= b.base_date
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
    JOIN trading_days t
      ON t.trade_offset = o.offset_days
    JOIN price_dates pd
      ON pd.parsed_time = t.traded_date
    JOIN `{full_table}` p
      ON p.time = pd.original_time
    ORDER BY t.base_date, o.offset_days
    """
    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("dates", "DATE", dates)]
        ),
    )
    return list(job.result())
