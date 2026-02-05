import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from hashlib import md5

sys.path.append("/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER")

from processor.news_processor import NewsProcessor
from processor.embedder import TitanEmbedder
from processor.bigquery.uploader import upload_processed_news_rows
from processor.bigquery.entity_triple_uploader import upload_entities_and_triples_rows


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


def _keyword_of(article):
    return article.get("key_word") or article.get("keyword") or ""


def _compute_mdhash_id(content: str, prefix: str) -> str:
    return prefix + md5(content.encode()).hexdigest()


def _pick_candidates(raw_items, keyword, existing_ids, processor, need):
    candidates = []
    for art in raw_items:
        if _keyword_of(art) != keyword:
            continue
        if art.get("id") in existing_ids:
            continue
        score = processor.calculate_heuristic_score(art)
        candidates.append((score, art))

    candidates.sort(key=lambda x: x[0], reverse=True)
    if len(candidates) <= need:
        return [art for _, art in candidates]
    return [art for _, art in candidates[:need]]


def _embed_and_collect(articles, embedder, existing_entity_ids, existing_triple_ids):
    new_entities = []
    new_triples = []

    for art in articles:
        text_to_embed = f"{art.get('title','')}\n\n{art.get('description','')}"
        art_vec = embedder.generate_embedding(text_to_embed, dimensions=512)
        art["article_embedding"] = art_vec

        for entity in art.get("named_entities", []) or []:
            entity_text = str(entity).strip()
            if not entity_text:
                continue
            entity_id = _compute_mdhash_id(entity_text, prefix="entity-")
            if entity_id in existing_entity_ids:
                continue
            vector = embedder.generate_embedding(entity_text, dimensions=1024)
            if vector:
                new_entities.append(
                    {"hash_id": entity_id, "entity_text": entity_text, "embedding": vector}
                )
                existing_entity_ids.add(entity_id)

        for triple in art.get("triples", []) or []:
            triple_text = str(triple).strip()
            if not triple_text:
                continue
            triple_id = _compute_mdhash_id(triple_text, prefix="triple-")
            if triple_id in existing_triple_ids:
                continue
            vector = embedder.generate_embedding(triple_text, dimensions=1024)
            if vector:
                new_triples.append(
                    {"hash_id": triple_id, "triple_text": triple_text, "embedding": vector}
                )
                existing_triple_ids.add(triple_id)

    return new_entities, new_triples


def main():
    parser = argparse.ArgumentParser(description="Fill shortfalls per keyword and upload to BigQuery.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--min-per-keyword", type=int, default=2)
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY is not set")

    processor = NewsProcessor(api_key=openai_key)
    embedder = TitanEmbedder(region_name="us-east-1")

    final_items = _load_json(FINAL_PATH)
    final_ids = {it.get("id") for it in final_items if it.get("id")}

    entity_json_path = os.path.join(PROCESSED_DIR, "entity.json")
    triple_json_path = os.path.join(PROCESSED_DIR, "triple.json")
    existing_entities = _load_json(entity_json_path)
    existing_triples = _load_json(triple_json_path)
    existing_entity_ids = {e.get("hash_id") for e in existing_entities if e.get("hash_id")}
    existing_triple_ids = {t.get("hash_id") for t in existing_triples if t.get("hash_id")}

    total_added = 0
    total_entities = []
    total_triples = []
    total_articles = []

    for d in _iter_dates(args.start_date, args.end_date):
        ymd = d.strftime("%Y%m%d")
        raw_path = os.path.join(RAW_DIR, f"news_backfill_{ymd}.json")
        processed_path = os.path.join(PROCESSED_DIR, f"processed_news_{ymd}.json")

        raw_items = _load_json(raw_path)
        if not raw_items:
            continue
        processed_items = _load_json(processed_path)

        counts = {k: 0 for k in KEYWORDS}
        processed_ids = {it.get("id") for it in processed_items if it.get("id")}
        for it in processed_items:
            kw = _keyword_of(it)
            if kw in counts:
                counts[kw] += 1

        added_today = []
        for kw in KEYWORDS:
            need = max(0, args.min_per_keyword - counts.get(kw, 0))
            if need == 0:
                continue
            picks = _pick_candidates(raw_items, kw, processed_ids, processor, need)
            for art in picks:
                art = dict(art)
                llm_data = processor.call_llm_extractor(art)
                art["filter_status"] = "T"
                art["named_entities"] = llm_data.get("named_entities", [])
                art["triples"] = llm_data.get("triples", [])
                art["fill_mode"] = "relaxed"
                art["fill_reason"] = f"min_per_keyword={args.min_per_keyword}"
                processed_items.append(art)
                processed_ids.add(art.get("id"))
                added_today.append(art)

        if added_today:
            _save_json(processed_path, processed_items)
            for art in added_today:
                if art.get("id") not in final_ids:
                    final_items.append(art)
                    final_ids.add(art.get("id"))
            new_entities, new_triples = _embed_and_collect(
                added_today, embedder, existing_entity_ids, existing_triple_ids
            )
            total_entities.extend(new_entities)
            total_triples.extend(new_triples)
            total_articles.extend(added_today)
            total_added += len(added_today)
            print(f"{ymd}: added {len(added_today)}")

    if total_added:
        _save_json(FINAL_PATH, final_items)
        if total_entities:
            _save_json(entity_json_path, existing_entities + total_entities)
        if total_triples:
            _save_json(triple_json_path, existing_triples + total_triples)

        upload_processed_news_rows(total_articles)
        upload_entities_and_triples_rows(total_entities, total_triples)

    print(f"Done. total_added={total_added}")


if __name__ == "__main__":
    main()
