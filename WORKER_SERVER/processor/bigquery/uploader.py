import argparse
import json
import os
from datetime import datetime, timedelta

from google.cloud import bigquery


def _load_schema(schema_path):
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_json = json.load(f)
    schema = []
    for field in schema_json:
        schema.append(
            bigquery.SchemaField(
                name=field.get("name"),
                field_type=field.get("type"),
                mode=field.get("mode", "NULLABLE"),
                description=field.get("description"),
            )
        )
    return schema


def _iter_date_range(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    for i in range((end - start).days + 1):
        yield start + timedelta(days=i)


def upload_processed_news(
    start_date="2026-01-13",
    end_date="2026-01-28",
    base_dir="/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER/data/processed",
    schema_path="/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER/processor/bigquery/schema.json",
):
    dataset_id = os.getenv("BIGQUERY_DATASET_ID")
    table_id = os.getenv("BIGQUERY_TABLE_ID")
    project_id = os.getenv("BIGQUERY_PROJECT_ID") or os.getenv("VERTEX_AI_PROJECT_ID")

    if not dataset_id or not table_id:
        raise ValueError("BIGQUERY_DATASET_ID and BIGQUERY_TABLE_ID must be set")

    if project_id:
        client = bigquery.Client(project=project_id)
    else:
        client = bigquery.Client()

    full_table_id = f"{client.project}.{dataset_id}.{table_id}"
    schema = None
    try:
        table = client.get_table(full_table_id)
        schema = table.schema
    except Exception:
        schema = _load_schema(schema_path)

    rows = []
    for d in _iter_date_range(start_date, end_date):
        path = os.path.join(base_dir, f"processed_news_{d.strftime('%Y%m%d')}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
        if isinstance(data, list):
            rows.extend(data)

    if not rows:
        print("No rows to upload.")
        return 0

    # Load into temp table then MERGE to avoid duplicates by id
    temp_table_id = f"{dataset_id}.temp_processed_news_{int(datetime.utcnow().timestamp())}"
    temp_full_table_id = f"{client.project}.{temp_table_id}"

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition="WRITE_TRUNCATE",
        ignore_unknown_values=True,
    )
    load_job = client.load_table_from_json(rows, temp_full_table_id, job_config=job_config)
    load_job.result()

    columns = [field.name for field in schema]
    if "id" not in columns:
        raise ValueError("Target table must have 'id' column for deduplication.")

    cols_csv = ", ".join([f"`{c}`" for c in columns])
    cols_values = ", ".join([f"S.`{c}`" for c in columns])

    merge_sql = f"""
    MERGE `{full_table_id}` T
    USING `{temp_full_table_id}` S
    ON T.id = S.id
    WHEN NOT MATCHED THEN
      INSERT ({cols_csv}) VALUES ({cols_values})
    """
    client.query(merge_sql).result()

    client.delete_table(temp_full_table_id, not_found_ok=True)

    print(f"Uploaded {len(rows)} rows to {full_table_id} (dedup by id)")
    return len(rows)


def _parse_args():
    parser = argparse.ArgumentParser(description="Upload processed news JSON to BigQuery.")
    parser.add_argument("--start-date", default="2026-01-13", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default="2026-01-28", help="YYYY-MM-DD")
    parser.add_argument("--base-dir", default="/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER/data/processed")
    parser.add_argument("--schema-path", default="/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER/processor/bigquery/schema.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    upload_processed_news(
        start_date=args.start_date,
        end_date=args.end_date,
        base_dir=args.base_dir,
        schema_path=args.schema_path,
    )
