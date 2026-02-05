import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from hashlib import md5

sys.path.append("/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER")

from processor.news_processor import NewsProcessor
from processor.embedder import TitanEmbedder
from processor.bigquery.uploader import upload_processed_news
from processor.bigquery.entity_triple_uploader import upload_entities_and_triples


DATA_DIR = "/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER/data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


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


def _compute_mdhash_id(content: str, prefix: str) -> str:
    return prefix + md5(content.encode()).hexdigest()


def _embed_processed_file(processed_path, entity_json_path, triple_json_path):
    if not os.path.exists(processed_path):
        print(f"No processed file: {processed_path}")
        return 0, 0, 0

    all_articles = _load_json(processed_path)
    if not all_articles:
        return 0, 0, 0

    embedder = TitanEmbedder(region_name="us-east-1")
    newly_embedded_articles = []
    new_entities = []
    new_triples = []

    existing_entities = _load_json(entity_json_path)
    existing_triples = _load_json(triple_json_path)
    existing_entity_ids = {e.get("hash_id") for e in existing_entities if e.get("hash_id")}
    existing_triple_ids = {t.get("hash_id") for t in existing_triples if t.get("hash_id")}

    for art in all_articles:
        if art.get("article_embedding") is None:
            text_to_embed = f"{art.get('title','')}\n\n{art.get('description','')}"
            vector = embedder.generate_embedding(text_to_embed, dimensions=512)
            if vector:
                art["article_embedding"] = vector
                newly_embedded_articles.append(art)

    for art in newly_embedded_articles:
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

    if newly_embedded_articles:
        with open(processed_path, "w", encoding="utf-8") as f:
            json.dump(all_articles, f, indent=4, ensure_ascii=False)

    if new_entities:
        with open(entity_json_path, "w", encoding="utf-8") as f:
            json.dump(existing_entities + new_entities, f, indent=4, ensure_ascii=False)

    if new_triples:
        with open(triple_json_path, "w", encoding="utf-8") as f:
            json.dump(existing_triples + new_triples, f, indent=4, ensure_ascii=False)

    return len(newly_embedded_articles), len(new_entities), len(new_triples)


def main():
    parser = argparse.ArgumentParser(description="Process/Embed/Upload backfill range.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY is not set")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    processor = NewsProcessor(api_key=openai_key)

    entity_json_path = os.path.join(PROCESSED_DIR, "entity.json")
    triple_json_path = os.path.join(PROCESSED_DIR, "triple.json")

    for d in _iter_dates(args.start_date, args.end_date):
        ymd = d.strftime("%Y%m%d")
        raw_path = os.path.join(RAW_DIR, f"news_backfill_{ymd}.json")
        processed_path = os.path.join(PROCESSED_DIR, f"processed_news_{ymd}.json")
        if not os.path.exists(raw_path):
            print(f"[Skip] missing raw: {raw_path}")
            continue

        processor.process_json_file(input_path=raw_path, output_path=processed_path)
        emb_n, ent_n, tri_n = _embed_processed_file(
            processed_path, entity_json_path, triple_json_path
        )
        print(f"[{ymd}] embedded_articles={emb_n} entities={ent_n} triples={tri_n}")

    # Upload all processed files in range
    upload_processed_news(start_date=args.start_date, end_date=args.end_date)
    upload_entities_and_triples()


if __name__ == "__main__":
    main()
