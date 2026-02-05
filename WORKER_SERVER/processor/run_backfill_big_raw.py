import argparse
import os
import sys

sys.path.append("/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER")

from crawler.backfill_crawler import fetch_and_standardize_backfill


DATA_DIR = "/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER/data"
RAW_DIR = os.path.join(DATA_DIR, "raw")


def main():
    parser = argparse.ArgumentParser(description="Run backfill once and save a single big raw file.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output", default="news_backfill_big.json")
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    output_path = os.path.join(RAW_DIR, args.output)
    print(f"[Backfill Big] {args.start_date} ~ {args.end_date} -> {output_path}")
    fetch_and_standardize_backfill(
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
