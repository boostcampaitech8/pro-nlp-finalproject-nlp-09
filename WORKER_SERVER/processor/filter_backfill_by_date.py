import argparse
import json
import os
from collections import defaultdict
from datetime import datetime


RAW_DIR = "/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER/data/raw"


def _parse_date(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except Exception:
        return None


def _load_json(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def main():
    parser = argparse.ArgumentParser(description="Filter backfill raw files by publish_date range.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--input-prefix", default="news_backfill_", help="Raw file prefix to scan")
    args = parser.parse_args()

    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    buckets = defaultdict(list)

    for name in os.listdir(RAW_DIR):
        if not (name.startswith(args.input_prefix) and name.endswith(".json")):
            continue
        items = _load_json(os.path.join(RAW_DIR, name))
        for it in items:
            d = _parse_date(it.get("publish_date"))
            if not d:
                continue
            if start <= d <= end:
                buckets[d.strftime("%Y%m%d")].append(it)

    written = 0
    for ymd, items in buckets.items():
        out_path = os.path.join(RAW_DIR, f"news_backfill_{ymd}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=4)
        written += 1

    print(f"Done. days_written={written} total_days_with_data={len(buckets)}")


if __name__ == "__main__":
    main()
