import os
import json
from news_processor import NewsProcessor # 기존 클래스
from embedder import TitanEmbedder        # 새로 만든 임베더

class OptimizedNewsProcessor(NewsProcessor):
    def __init__(self, api_key):
        super().__init__(api_key=api_key)
        self.embedder = TitanEmbedder() # Titan v2 모델 로드

    def process_json_file(self, input_path, output_path):
        """뉴스 가공 후, 새롭게 추가된 T 뉴스들만 리스트로 반환합니다."""
        # 1. 기존 가공 완료된 데이터 로드 (중복 체크용)
        final_results = []
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                final_results = json.load(f)
        
        processed_ids = {art['id'] for art in final_results}
        
        # 2. 이번에 수집된 원본(raw) 뉴스 로드
        with open(input_path, 'r', encoding='utf-8') as f:
            new_raw_data = json.load(f)

        newly_processed_articles = [] # 🌟 빅쿼리로 쏠 새 뉴스 바구니

        for art in new_raw_data:
            if art['id'] in processed_ids:
                continue # 이미 처리한 건 패스

            # LLM 가공 (T/F 필터링, Entity 추출 등 수행)
            llm_data = self.llm_filter_and_extract(art) 
            
            if llm_data.get('filter_status') == 'T':
                # 🟢 가공 직후 바로 임베딩 생성 (Title + Description)
                embed_text = f"{art['title']}\n\n{art.get('description', '')}"
                art['article_embedding'] = self.embedder.generate_embedding(embed_text)
                
                art.update(llm_data)
                final_results.append(art)
                newly_processed_articles.append(art) # 새 바구니에 담기

        # 3. 전체 금고 업데이트 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=4, ensure_ascii=False)

        # 🌟 '이번 회차에 새로 탄생한' 기사들만 반환!
        return newly_processed_articles

# --- 실행부 ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")

processor = OptimizedNewsProcessor(api_key=api_key)
# 이제 result에는 새로 추가된 기사들만 담깁니다.
new_additions = processor.process_json_file("collected_news.json", "final_processed_news.json")