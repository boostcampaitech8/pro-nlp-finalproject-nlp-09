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
    if not hash_ids:
        return []
    dataset = os.getenv("BIGQUERY_DATASET_ID", "tilda")
    table = os.getenv("TRIPLE_MAP_TABLE", "triple_article_map")
    full_table = f"{client.project}.{dataset}.{table}"
    query = f"""
    SELECT DISTINCT hash_id, article_id, publish_date
    FROM `{full_table}`
    WHERE hash_id IN UNNEST(@hash_ids)
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
    trading_days AS (
      SELECT
        b.base_date,
        p.time AS traded_date,
        ROW_NUMBER() OVER (PARTITION BY b.base_date ORDER BY p.time) - 1 AS trade_offset
      FROM base_dates b
      JOIN (
        SELECT DISTINCT time
        FROM `{full_table}`
      ) p
        ON p.time >= b.base_date
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
    JOIN `{full_table}` p
      ON p.time = t.traded_date
    ORDER BY t.base_date, o.offset_days
    """
    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("dates", "DATE", dates)]
        ),
    )
    return list(job.result())
