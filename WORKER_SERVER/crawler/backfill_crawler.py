import argparse
import json
import os
from datetime import datetime, timedelta

import requests

try:
    from .config import NEWS_API_KEY, TARGET_QUERIES
    from .utils import generate_md5_id, format_date
except ImportError:
    from config import NEWS_API_KEY, TARGET_QUERIES
    from utils import generate_md5_id, format_date


def _iso_day_start(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%Y-%m-%dT00:00:00Z")


def _iso_day_end(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)
    return dt.strftime("%Y-%m-%dT23:59:59Z")


def _ymd(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%Y-%m-%d")


def _get_newsapi_key():
    api_key = os.getenv("NEWS_API_KEY")
    if api_key:
        return api_key
    try:
        from .config import NEWS_API_KEY as CONFIG_NEWS_API_KEY  # type: ignore
        return CONFIG_NEWS_API_KEY
    except Exception:
        return None


def _get_gnews_key():
    api_key = os.getenv("GNEWS_API_KEY")
    if api_key:
        return api_key
    try:
        from .config import GNEWS_API_KEY as CONFIG_GNEWS_API_KEY  # type: ignore
        return CONFIG_GNEWS_API_KEY
    except Exception:
        return "a971051f1af9d6f09ff430a38c11c1e4"


def fetch_and_standardize_backfill(start_date, end_date=None, output_path=None, page_size=50):
    """
    Backfill collector: fetch articles between start_date and end_date (inclusive), day-range scoped.
    Dates must be YYYY-MM-DD.
    """
    if not end_date:
        end_date = start_date

    unique_news_dict = {}

    from_iso = _iso_day_start(start_date)
    to_iso = _iso_day_end(end_date)
    from_ymd = _ymd(start_date)
    to_ymd = _ymd(end_date)

    def _normalize_keyword(query, title, description, content):
        text = f"{query} {title or ''} {description or ''} {content or ''}".lower()
        if "usda" in text or "united states department of agriculture" in text:
            return "\"United States Department of Agriculture\" OR USDA"
        if "soybean" in text or "soybeans" in text:
            return "soybean AND (price OR demand OR supply OR inventory)"
        if "wheat" in text:
            return "wheat AND (price OR demand OR supply OR inventory)"
        if "corn" in text or "maize" in text:
            return "corn AND (price OR demand OR supply OR inventory)"
        return query

    def _fetch_from_newsapi():
        base_url = "https://newsapi.org/v2/everything"
        api_key = _get_newsapi_key()
        if not api_key:
            raise ValueError("NEWS_API_KEY is not set")

        for query in TARGET_QUERIES:
            page = 1
            total_results = None

            while True:
                params = {
                    "q": query,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "from": from_iso,
                    "to": to_iso,
                    "apiKey": api_key,
                    "pageSize": page_size,
                    "page": page,
                }

                try:
                    print(f"[NewsAPI] query='{query}' page={page} range={from_iso}~{to_iso}")
                    response = requests.get(base_url, params=params, timeout=120)
                    if response.status_code != 200:
                        print(f"Error fetching {query} (page {page}): {response.status_code}")
                        break

                    payload = response.json()
                    if payload.get("status") == "error":
                        print(f"Error fetching {query} (page {page}): {payload.get('message')}")
                        break

                    articles = payload.get("articles", [])
                    total_results = payload.get("totalResults", 0)

                    if not articles:
                        break

                    for art in articles:
                        url = art.get("url")
                        if not url:
                            continue
                        article_id = generate_md5_id(url)

                        if article_id in unique_news_dict:
                            continue

                        published_at = art.get("publishedAt")
                        if not published_at:
                            continue
                        standard_row = {
                            "id": article_id,
                            "title": art.get("title"),
                            "doc_url": url,
                            "all_text": art.get("content")[:1500] if art.get("content") else "",
                            "authors": art.get("author", "None"),
                            "publish_date": format_date(published_at),
                            "published_at_raw": published_at,
                            "meta_site_name": art.get("source", {}).get("name"),
                            "description": art.get("description"),
                            "key_word": _normalize_keyword(query, art.get("title"), art.get("description"), art.get("content")),
                            "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "filter_status": "F",
                            "named_entities": [],
                            "triples": [],
                            "article_embedding": None,
                        }
                        unique_news_dict[article_id] = standard_row

                    if total_results is not None:
                        if page * page_size >= total_results:
                            break
                    if page >= 100:
                        break

                    page += 1

                except Exception as e:
                    print(f"Error fetching {query} (page {page}): {e}")
                    break

    def _fetch_from_gnews():
        base_url = "https://gnews.io/api/v4/search"
        api_key = _get_gnews_key()
        if not api_key:
            raise ValueError("GNEWS_API_KEY is not set")

        for query in TARGET_QUERIES:
            page = 1
            total_results = None

            while True:
                params = {
                    "q": query,
                    "lang": "en",
                    "from": from_ymd,
                    "to": to_ymd,
                    "max": page_size,
                    "page": page,
                    "token": api_key,
                }

                try:
                    print(f"[GNews] query='{query}' page={page} range={from_ymd}~{to_ymd}")
                    response = requests.get(base_url, params=params, timeout=120)
                    if response.status_code != 200:
                        print(f"Error fetching {query} (page {page}): {response.status_code}")
                        break

                    payload = response.json()
                    if payload.get("errors"):
                        print(f"Error fetching {query} (page {page}): {payload.get('errors')}")
                        break

                    articles = payload.get("articles", [])
                    total_results = payload.get("totalArticles", 0)

                    if not articles:
                        break

                    for art in articles:
                        url = art.get("url")
                        if not url:
                            continue
                        article_id = generate_md5_id(url)

                        if article_id in unique_news_dict:
                            continue

                        published_at = art.get("publishedAt")
                        if not published_at:
                            continue
                        standard_row = {
                            "id": article_id,
                            "title": art.get("title"),
                            "doc_url": url,
                            "all_text": art.get("content")[:1500] if art.get("content") else "",
                            "authors": art.get("author", "None"),
                            "publish_date": format_date(published_at),
                            "published_at_raw": published_at,
                            "meta_site_name": art.get("source", {}).get("name"),
                            "description": art.get("description"),
                            "key_word": _normalize_keyword(query, art.get("title"), art.get("description"), art.get("content")),
                            "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "filter_status": "F",
                            "named_entities": [],
                            "triples": [],
                            "article_embedding": None,
                        }
                        unique_news_dict[article_id] = standard_row

                    if total_results is not None:
                        if page * page_size >= total_results:
                            break
                    if page >= 100:
                        break

                    page += 1

                except Exception as e:
                    print(f"Error fetching {query} (page {page}): {e}")
                    break

    if _get_gnews_key():
        _fetch_from_gnews()
    else:
        _fetch_from_newsapi()

    results = list(unique_news_dict.values())
    # NOTE: No publish_date filtering here (big raw collection mode)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"🚀 저장 완료: {output_path} ({len(results)}건)")

    return results


def _parse_args():
    parser = argparse.ArgumentParser(description="Backfill news collector by date range.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", help="YYYY-MM-DD (optional, defaults to start-date)")
    parser.add_argument("--output", help="Output JSON file path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    fetch_and_standardize_backfill(
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.output,
    )
