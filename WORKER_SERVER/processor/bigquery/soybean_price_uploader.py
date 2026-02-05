import argparse
import json
import os
from datetime import datetime

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


def _get_bq_client():
    project_id = os.getenv("BIGQUERY_PROJECT_ID") or os.getenv("VERTEX_AI_PROJECT_ID")
    if project_id:
        return bigquery.Client(project=project_id)
    return bigquery.Client()


def _compute_ema(series, period):
    if not series:
        return []
    alpha = 2 / (period + 1)
    ema_values = [series[0]]
    for price in series[1:]:
        ema_values.append(alpha * price + (1 - alpha) * ema_values[-1])
    return ema_values


def _fetch_tradingview_history(symbol, exchange, bars):
    try:
        from tvDatafeed import TvDatafeed, Interval
    except Exception as e:
        raise ImportError("tvDatafeed is required. Install with `pip install tvdatafeed`.") from e

    username = os.getenv("TV_USERNAME")
    password = os.getenv("TV_PASSWORD")
    tv = TvDatafeed(username=username, password=password)
    return tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_daily, n_bars=bars)


def upload_soybean_price_for_date(
    target_date,
    ema_period=20,
    symbol="ZS1!",
    exchange="CBOT",
    dataset_id="tilda",
    table_id="soybean_price",
    schema_path="/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER/processor/bigquery/corn_price_schema.json",
):
    client = _get_bq_client()
    full_table_id = f"{client.project}.{dataset_id}.{table_id}"

    bars = max(ema_period * 3, 60)
    df = _fetch_tradingview_history(symbol, exchange, bars)
    if df is None or df.empty:
        print("No data from TradingView.")
        return 0

    df = df.sort_index()
    df["ema"] = _compute_ema(df["close"].tolist(), ema_period)

    target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
    day_row = df[df.index.date == target_dt]
    if day_row.empty:
        print(f"No data for {target_date}")
        return 0

    row = day_row.iloc[-1]
    payload = [
        {
            "time": target_date,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "EMA": float(row["ema"]),
            "Volume": int(row["volume"]),
        }
    ]

    schema = _load_schema(schema_path)
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND")
    load_job = client.load_table_from_json(payload, full_table_id, job_config=job_config)
    load_job.result()

    print(f"Uploaded 1 row for {target_date} to {full_table_id}")
    return 1


def _parse_args():
    parser = argparse.ArgumentParser(description="Upload daily soybean price to BigQuery.")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--ema-period", type=int, default=20)
    parser.add_argument("--symbol", default="ZS1!")
    parser.add_argument("--exchange", default="CBOT")
    parser.add_argument("--dataset-id", default="tilda")
    parser.add_argument("--table-id", default="soybean_price")
    parser.add_argument(
        "--schema-path",
        default="/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER/processor/bigquery/corn_price_schema.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    upload_soybean_price_for_date(
        target_date=args.date,
        ema_period=args.ema_period,
        symbol=args.symbol,
        exchange=args.exchange,
        dataset_id=args.dataset_id,
        table_id=args.table_id,
        schema_path=args.schema_path,
    )
