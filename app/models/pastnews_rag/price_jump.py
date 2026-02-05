"""
과거 뉴스 사례 분석을 위한 가격 변동 조회 모듈
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
    hash_ids로 기사 정보와 발행일을 조회합니다.
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


def fetch_prices_for_dates(client, dates, commodity: str = "corn"):
    """
    특정 날짜들 전후의 품목별 가격 변동을 조회합니다.
    """
    if not dates:
        return []
    dataset = os.getenv("BIGQUERY_DATASET_ID", "tilda")
    # 품목에 맞는 가격 테이블 선택
    table = f"{commodity}_price"
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