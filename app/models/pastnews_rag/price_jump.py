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
    offsets AS (
      SELECT base_date, base_date as target_date, 0 as offset_days FROM base_dates
      UNION ALL SELECT base_date, DATE_ADD(base_date, INTERVAL 1 DAY), 1 FROM base_dates
      UNION ALL SELECT base_date, DATE_ADD(base_date, INTERVAL 3 DAY), 3 FROM base_dates
    ),
    next_trading AS (
      SELECT
        o.base_date,
        o.offset_days,
        MIN(p.time) AS traded_date
      FROM offsets o
      JOIN `{full_table}` p
        ON p.time >= o.target_date
      GROUP BY o.base_date, o.offset_days
    )
    SELECT
      n.base_date,
      n.offset_days,
      n.traded_date,
      p.close
    FROM next_trading n
    JOIN `{full_table}` p
      ON p.time = n.traded_date
    ORDER BY n.base_date, n.offset_days
    """
    job = client.query(
        query,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("dates", "DATE", dates)]
        ),
    )
    return list(job.result())
