import argparse
import os
import sys
from datetime import datetime, timedelta

sys.path.append("/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER")

from crawler.backfill_crawler import fetch_and_standardize_backfill


DATA_DIR = "/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER/data"
RAW_DIR = os.path.join(DATA_DIR, "raw")


def _iter_dates(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(description="Run backfill once for a date range (daily raw files).")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)

    for d in _iter_dates(args.start_date, args.end_date):
        ymd = d.strftime("%Y%m%d")
        output_path = os.path.join(RAW_DIR, f"news_backfill_{ymd}.json")
        print(f"[Backfill] {d} -> {output_path}")
        fetch_and_standardize_backfill(
            start_date=d.strftime("%Y-%m-%d"),
            end_date=d.strftime("%Y-%m-%d"),
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
