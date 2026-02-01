import argparse
import json
import os

from google.cloud import bigquery


def _get_bq_client():
    project_id = os.getenv("BIGQUERY_PROJECT_ID") or os.getenv("VERTEX_AI_PROJECT_ID")
    if project_id:
        return bigquery.Client(project=project_id)
    return bigquery.Client()


def _load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []
    return data if isinstance(data, list) else []


def _coerce_embedding(row):
    emb = row.get("embedding")
    if emb is None:
        return None
    if isinstance(emb, list):
        return [float(x) for x in emb]
    return None


def _upload_rows(client, dataset_id, table_id, rows, schema):
    if not rows:
        print(f"No rows to upload for {table_id}.")
        return 0
    full_table_id = f"{client.project}.{dataset_id}.{table_id}"
    temp_table_id = f"{dataset_id}.temp_{table_id}_{int(__import__('time').time())}"
    temp_full_table_id = f"{client.project}.{temp_table_id}"
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition="WRITE_TRUNCATE",
        ignore_unknown_values=True,
    )
    load_job = client.load_table_from_json(rows, temp_full_table_id, job_config=job_config)
    load_job.result()

    columns = [field.name for field in schema]
    if "hash_id" not in columns:
        raise ValueError("Target table must have 'hash_id' column for deduplication.")
    cols_csv = ", ".join([f"`{c}`" for c in columns])
    cols_values = ", ".join([f"S.`{c}`" for c in columns])
    merge_sql = f"""
    MERGE `{full_table_id}` T
    USING `{temp_full_table_id}` S
    ON T.hash_id = S.hash_id
    WHEN NOT MATCHED THEN
      INSERT ({cols_csv}) VALUES ({cols_values})
    """
    client.query(merge_sql).result()
    client.delete_table(temp_full_table_id, not_found_ok=True)

    print(f"Uploaded {len(rows)} rows to {full_table_id} (dedup by hash_id)")
    return len(rows)


def upload_entities_and_triples(
    base_dir="/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER/data/processed",
    entities_filename="entity.json",
    triples_filename="triple.json",
    dataset_id=None,
    entities_table="news_article_entities",
    triples_table="news_article_triples",
):
    dataset_id = dataset_id or os.getenv("BIGQUERY_DATASET_ID")
    if not dataset_id:
        raise ValueError("BIGQUERY_DATASET_ID must be set")

    client = _get_bq_client()

    entities_schema = [
        bigquery.SchemaField("hash_id", "STRING"),
        bigquery.SchemaField("entity_text", "STRING"),
        bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED"),
    ]
    triples_schema = [
        bigquery.SchemaField("hash_id", "STRING"),
        bigquery.SchemaField("triple_text", "STRING"),
        bigquery.SchemaField("embedding", "FLOAT", mode="REPEATED"),
    ]

    entities_path = os.path.join(base_dir, entities_filename)
    triples_path = os.path.join(base_dir, triples_filename)

    entities_raw = _load_json(entities_path)
    triples_raw = _load_json(triples_path)

    entities_rows = []
    for row in entities_raw:
        entities_rows.append(
            {
                "hash_id": row.get("hash_id"),
                "entity_text": row.get("entity_text"),
                "embedding": _coerce_embedding(row),
            }
        )

    triples_rows = []
    for row in triples_raw:
        triples_rows.append(
            {
                "hash_id": row.get("hash_id"),
                "triple_text": row.get("triple_text"),
                "embedding": _coerce_embedding(row),
            }
        )

    uploaded_entities = _upload_rows(
        client, dataset_id, entities_table, entities_rows, entities_schema
    )
    uploaded_triples = _upload_rows(
        client, dataset_id, triples_table, triples_rows, triples_schema
    )
    return uploaded_entities, uploaded_triples


def _parse_args():
    parser = argparse.ArgumentParser(description="Upload entity/triple JSON to BigQuery.")
    parser.add_argument(
        "--base-dir",
        default="/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER/data/processed",
    )
    parser.add_argument("--entities-file", default="entity.json")
    parser.add_argument("--triples-file", default="triple.json")
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--entities-table", default="news_article_entities")
    parser.add_argument("--triples-table", default="news_article_triples")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    upload_entities_and_triples(
        base_dir=args.base_dir,
        entities_filename=args.entities_file,
        triples_filename=args.triples_file,
        dataset_id=args.dataset_id,
        entities_table=args.entities_table,
        triples_table=args.triples_table,
    )
