import requests
import os
import json
from datetime import datetime

try:
    from .config import NEWS_API_KEY, TARGET_QUERIES
    from .utils import generate_md5_id, format_date
except ImportError:
    from config import NEWS_API_KEY, TARGET_QUERIES
    from utils import generate_md5_id, format_date

def fetch_and_standardize(output_path=None):
    base_url = "https://newsapi.org/v2/everything"
    unique_news_dict = {}

    for query in TARGET_QUERIES:
        params = {
            "q": query,
            "language": "en",
            "sortBy": 'publishedAt',
            "apiKey": NEWS_API_KEY,
            "pageSize": 50  # 확장성을 위해 페이지 사이즈를 좀 더 키웠습니다.
        }
        
        try:
            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                continue
                
            articles = response.json().get('articles', [])
            
            for art in articles:
                article_id = generate_md5_id(art['url'])
                
                # 1단계 런타임 중복 제거 (쿼리 간 중복)
                if article_id in unique_news_dict:
                    continue

                standard_row = {
                    "id": article_id,
                    "title": art.get('title'),
                    "doc_url": art.get('url'),
                    "all_text": art.get('content')[:1500] if art.get('content') else "",
                    "authors": art.get('author', "None"),
                    "publish_date": format_date(art.get('publishedAt')),
                    "meta_site_name": art.get('source', {}).get('name'),
                    "description": art.get('description'),
                    "key_word": query,
                    "collected_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), # 수집 시간 기록
                    "filter_status": "F",
                    "named_entities": [],
                    "triples": [],
                    "article_embedding": None
                }
                unique_news_dict[article_id] = standard_row
                
        except Exception as e:
            print(f"Error fetching {query}: {e}")

    results = list(unique_news_dict.values())

    # Airflow에서 관리하기 쉽도록 output_path 처리
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"🚀 저장 완료: {output_path} ({len(results)}건)")
    
    return results

if __name__ == "__main__":
    # 로컬 테스트용: 현재 시간 기반 파일명 생성
    current_time = datetime.now().strftime('%Y%m%d_%H')
    test_path = f"data/raw/news_{current_time}.json"
    fetch_and_standardize(output_path=test_path)