import argparse
import json
import os
from collections import defaultdict
from datetime import datetime


def _parse_date(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Split one big raw file into daily backfill files by publish_date.")
    parser.add_argument("--input", required=True, help="Path to big raw json")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--out-dir", default="/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER/data/raw")
    args = parser.parse_args()

    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    with open(args.input, "r", encoding="utf-8") as f:
        items = json.load(f)

    buckets = defaultdict(list)
    for it in items:
        d = _parse_date(it.get("publish_date"))
        if not d:
            continue
        if start <= d <= end:
            buckets[d.strftime("%Y%m%d")].append(it)

    os.makedirs(args.out_dir, exist_ok=True)
    for ymd, rows in buckets.items():
        out_path = os.path.join(args.out_dir, f"news_backfill_{ymd}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=4)

    print(f"Done. days_written={len(buckets)}")


if __name__ == "__main__":
    main()
