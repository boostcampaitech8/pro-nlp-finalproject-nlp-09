import os
from datetime import timedelta

from google.cloud import bigquery


def _get_bq_client():
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


def run_step3_example():
    # Step 3 예시: Step 2 결과 hash_id 5개를 그대로 사용
    hash_ids = [
        "triple-6d1c35f599f3ff5446d1b1c5741a4609",
        "triple-2239e3cc472d6a72904cad548f02d710",
        "triple-f5aecb3858cdd3b47c65f98ab585a300",
        "triple-3da92bf4ebd19f569f43401cae24a517",
        "triple-7b0a11d02283be13011966f40d8dfb31",
    ]

    client = _get_bq_client()
    mappings = fetch_article_dates(client, hash_ids)
    dates = [row.publish_date for row in mappings]
    prices = fetch_prices_for_dates(client, dates)

    print("Mappings (hash_id, article_id, publish_date):")
    for row in mappings[:10]:
        print(row.hash_id, row.article_id, row.publish_date)

    print("\nPrices (base_date, offset_days, traded_date, close):")
    for row in prices[:20]:
        print(row.base_date, row.offset_days, row.traded_date, row.close)


if __name__ == "__main__":
    run_step3_example()
