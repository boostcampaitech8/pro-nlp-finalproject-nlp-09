import argparse
import json
import os
from datetime import datetime, timedelta

from processor.news_processor import NewsProcessor


DATA_DIR = "/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER/data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FINAL_PATH = os.path.join(PROCESSED_DIR, "final_processed_news.json")

KEYWORDS = [
    "corn AND (price OR demand OR supply OR inventory)",
    "soybean AND (price OR demand OR supply OR inventory)",
    "wheat AND (price OR demand OR supply OR inventory)",
    "\"United States Department of Agriculture\" OR USDA",
]


def _iter_dates(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _load_json(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def _load_raw_for_date(date_str):
    backfill_path = os.path.join(RAW_DIR, f"news_backfill_{date_str.replace('-','')}.json")
    if os.path.exists(backfill_path):
        return _load_json(backfill_path)

    # fallback: hourly raw files
    prefix = f"news_{date_str.replace('-','')}_"
    items = []
    for name in os.listdir(RAW_DIR):
        if name.startswith(prefix) and name.endswith(".json"):
            items.extend(_load_json(os.path.join(RAW_DIR, name)))
    return items


def _keyword_of(article):
    return article.get("key_word") or article.get("keyword") or ""


def _ensure_min_per_keyword(processed_items, raw_items, processor, min_per_keyword=2):
    # index processed by keyword
    counts = {k: 0 for k in KEYWORDS}
    existing_ids = {it.get("id") for it in processed_items if it.get("id")}
    for it in processed_items:
        kw = _keyword_of(it)
        if kw in counts:
            counts[kw] += 1

    added = []
    for kw in KEYWORDS:
        need = max(0, min_per_keyword - counts.get(kw, 0))
        if need == 0:
            continue

        # candidates: same keyword, not already processed
        candidates = []
        for art in raw_items:
            if _keyword_of(art) != kw:
                continue
            if art.get("id") in existing_ids:
                continue
            score = processor.calculate_heuristic_score(art)
            candidates.append((score, art))

        # take top by heuristic score
        candidates.sort(key=lambda x: x[0], reverse=True)
        for score, art in candidates[:need]:
            art = dict(art)
            art["filter_status"] = "T"
            art["fill_mode"] = "relaxed"
            art["source_date"] = str(art.get("publish_date", ""))[:10]
            art["target_date"] = str(art.get("publish_date", ""))[:10]
            art["fill_reason"] = f"min_per_keyword={min_per_keyword}"
            processed_items.append(art)
            added.append(art)
            existing_ids.add(art.get("id"))
            counts[kw] = counts.get(kw, 0) + 1

    return added


def main():
    parser = argparse.ArgumentParser(description="Fill daily minimum news per keyword.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--min-per-keyword", type=int, default=2)
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY is not set")
    processor = NewsProcessor(api_key=openai_key)

    final_items = _load_json(FINAL_PATH)
    final_ids = {it.get("id") for it in final_items if it.get("id")}
    total_added = 0

    for d in _iter_dates(args.start_date, args.end_date):
        date_str = d.strftime("%Y-%m-%d")
        processed_path = os.path.join(PROCESSED_DIR, f"processed_news_{d.strftime('%Y%m%d')}.json")
        processed_items = _load_json(processed_path)

        raw_items = _load_raw_for_date(date_str)
        if not raw_items:
            continue

        added = _ensure_min_per_keyword(
            processed_items,
            raw_items,
            processor,
            min_per_keyword=args.min_per_keyword,
        )

        if added:
            _save_json(processed_path, processed_items)
            for art in added:
                if art.get("id") not in final_ids:
                    final_items.append(art)
                    final_ids.add(art.get("id"))
            total_added += len(added)
            print(f"{date_str}: added {len(added)}")

    _save_json(FINAL_PATH, final_items)
    print(f"Done. total_added={total_added}")


if __name__ == "__main__":
    main()
