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
    location = os.getenv("BIGQUERY_LOCATION", "US")
    full_table_id = f"{client.project}.{dataset_id}.{table_id}"
    print(f"[BQ] project={client.project} dataset={dataset_id} table={table_id} location={location}")
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition="WRITE_APPEND",
        ignore_unknown_values=True,
    )
    load_job = client.load_table_from_json(
        rows, full_table_id, job_config=job_config, location=location
    )
    load_job.result()

    print(f"Uploaded {len(rows)} rows to {full_table_id} (append)")
    return len(rows)


def upload_entities_and_triples(
    base_dir="/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER/data/processed",
    entities_filename="entity.json",
    triples_filename="triple.json",
    dataset_id=None,
    entities_table=None,
    triples_table=None,
):
    dataset_id = dataset_id or os.getenv("BIGQUERY_DATASET_ID", "tilda")
    if not dataset_id:
        raise ValueError("BIGQUERY_DATASET_ID must be set")
    entities_table = entities_table or os.getenv("ENTITIES_TABLE", "news_article_entities")
    triples_table = triples_table or os.getenv("TRIPLES_TABLE", "news_article_triples")

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


def upload_entities_and_triples_rows(
    entities_rows,
    triples_rows,
    dataset_id=None,
    entities_table=None,
    triples_table=None,
):
    dataset_id = dataset_id or os.getenv("BIGQUERY_DATASET_ID", "tilda")
    if not dataset_id:
        raise ValueError("BIGQUERY_DATASET_ID must be set")
    entities_table = entities_table or os.getenv("ENTITIES_TABLE", "news_article_entities")
    triples_table = triples_table or os.getenv("TRIPLES_TABLE", "news_article_triples")

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

    entities_rows = [
        {
            "hash_id": row.get("hash_id"),
            "entity_text": row.get("entity_text"),
            "embedding": _coerce_embedding(row),
        }
        for row in (entities_rows or [])
    ]

    triples_rows = [
        {
            "hash_id": row.get("hash_id"),
            "triple_text": row.get("triple_text"),
            "embedding": _coerce_embedding(row),
        }
        for row in (triples_rows or [])
    ]

    print(f"Preparing to upload {len(entities_rows)} entity rows and {len(triples_rows)} triple rows (XCom payload).")

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
