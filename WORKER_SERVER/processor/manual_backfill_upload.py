import json
import os
import sys
from hashlib import md5

sys.path.append("/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER")

from processor.embedder import TitanEmbedder
from processor.news_processor import NewsProcessor
from processor.bigquery.uploader import upload_processed_news_rows
from processor.bigquery.entity_triple_uploader import upload_entities_and_triples_rows


RAW_PATH = "/data/ephemeral/home/pro-nlp-finalproject-nlp-09/WORKER_SERVER/data/raw/news_20260202_15.json"

TARGET_TITLES = [
    "Landrith: FL Farm Bill leads way for agricultural security",
    "Subsidy allocation for food, fertiliser, fuel cut 5%",
    "FAO and Asian Development Bank advance $100 million",
    "How Bihar is attempting smoother ration delivery with new ‘Grain ATMs’",
    "Solar panels on land used for biofuels could power all cars and trucks electric",
]


def _compute_mdhash_id(content: str, prefix: str) -> str:
    return prefix + md5(content.encode()).hexdigest()


def _match_targets(article_title: str) -> bool:
    title = (article_title or "").strip()
    return any(t in title for t in TARGET_TITLES)


def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(RAW_PATH)

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY is not set")

    os.environ["BIGQUERY_DATASET_ID"] = os.getenv("BIGQUERY_DATASET_ID", "tilda")
    os.environ["BIGQUERY_TABLE_ID"] = os.getenv("BIGQUERY_TABLE_ID", "news_article")
    os.environ["ENTITIES_TABLE"] = os.getenv("ENTITIES_TABLE", "news_article_entities")
    os.environ["TRIPLES_TABLE"] = os.getenv("TRIPLES_TABLE", "news_article_triples")

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    selected = [it for it in raw_items if _match_targets(it.get("title"))]
    if not selected:
        print("No matching articles found.")
        return

    print(f"Selected {len(selected)} target articles.")

    processor = NewsProcessor(api_key=openai_key)
    embedder = TitanEmbedder(region_name="us-east-1")

    new_articles = []
    new_entities = []
    new_triples = []

    for art in selected:
        llm_data = processor.call_llm_extractor(art)
        if llm_data.get("filter_status") != "T":
            # Force include since user explicitly selected these titles
            llm_data["filter_status"] = "T"

        art.update(llm_data)

        text_to_embed = f"{art.get('title','')}\n\n{art.get('description','')}"
        vector = embedder.generate_embedding(text_to_embed, dimensions=512)
        art["article_embedding"] = vector
        new_articles.append(art)

        for entity in art.get("named_entities", []) or []:
            entity_text = str(entity).strip()
            if not entity_text:
                continue
            entity_id = _compute_mdhash_id(entity_text, prefix="entity-")
            ent_vec = embedder.generate_embedding(entity_text, dimensions=1024)
            if ent_vec:
                new_entities.append(
                    {"hash_id": entity_id, "entity_text": entity_text, "embedding": ent_vec}
                )

        for triple in art.get("triples", []) or []:
            triple_text = str(triple).strip()
            if not triple_text:
                continue
            triple_id = _compute_mdhash_id(triple_text, prefix="triple-")
            tri_vec = embedder.generate_embedding(triple_text, dimensions=1024)
            if tri_vec:
                new_triples.append(
                    {"hash_id": triple_id, "triple_text": triple_text, "embedding": tri_vec}
                )

    print(
        f"Prepared articles={len(new_articles)} entities={len(new_entities)} triples={len(new_triples)}"
    )

    upload_processed_news_rows(new_articles)
    upload_entities_and_triples_rows(new_entities, new_triples)


if __name__ == "__main__":
    main()
