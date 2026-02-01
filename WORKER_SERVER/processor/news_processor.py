import json
import os
import time
from openai import OpenAI

class NewsProcessor:
    def __init__(self, api_key):
        # 1. 키워드 점수제 설정
        self.commodities = ['corn', 'maize', 'wheat', 'soybean', 'soybeans', 'grain', 'grains', 'crop', 'crops']
        self.market = ['price', 'prices', 'demand', 'supply', 'inventory', 'stock', 'stocks', 'export', 'import', 'shipment', 'cargo', 'basis', 'futures', 'harvest', 'yield', 'acreage', 'planting']
        self.policy_climate = ['usda', 'united states department of agriculture', 'policy', 'tariff', 'subsidy', 'sanction', 'quota', 'regulation', 'climate', 'climate change', 'drought', 'flood', 'heatwave', 'el niño', 'la niña']
        self.exclude = ['corn palace', 'classic', 'tournament', 'basketball', 'football', 'match', 'game', 'deer', 'hummingbird', 'ferret', 'dog', 'cat', 'pet', 'wildlife', 'feeder', 'vaccine', 'cancer', 'detox', 'miracle', 'recipe', 'cooking', 'kitchen', 'how to', 'diy']
        
        # 2. OpenAI Client 생성
        self.client = OpenAI(api_key=api_key)
        
    def calculate_heuristic_score(self, article):
        """1차 규칙 필터: 점수 기반 노이즈 제거"""
        text = f"{article['title']} {article.get('description', '')}".lower()
        if any(e in text for e in self.exclude):
            return 0
        
        score = 0
        if any(c in text for c in self.commodities): score += 2
        if any(m in text for m in self.market): score += 1
        if any(pc in text for pc in self.policy_climate): score += 2
        return score

    def call_llm_extractor(self, article):
        """2차 LLM: GPT-4o-mini 기반 정밀 판별 및 트리플 추출"""
        prompt = f"""
        당신은 글로벌 농산물(옥수수, 대두, 밀) 시장 분석가입니다. 아래 기사를 분석하여 JSON 형식으로 응답하세요.

        [분석 지침]
        1. Relevance (filter_status): 옥수수, 대두, 밀의 가격/수급/정책/기상과 관련 있으면 "T", 아니면 "F".
        2. Named Entities: 국가, 기관(USDA 등), 작물명, 사건명 리스트.
        3. Triples: [주체, 동작, 결과] 뿐만 아니라, 사건의 원인이나 영향을 수치화할 수 있는 정보가 있다면 포함하세요.
        (예: ["러시아", "수출 중단", "밀 가격 상승 예상"])
        
        기사 제목: {article['title']}
        기사 내용: {article['all_text']}

        JSON 출력 형식:
        {{
            "filter_status": "T",
            "named_entities": ["entity1", "entity2"],
            "triples": [["Subject", "Predicate", "Object"]]
        }}
        """
        try:
            # GPT-4o-mini 호출 방식 (JSON 모드 지원)
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional analyst who extracts structured data in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" } # JSON 형식을 강제함
            )
            
            # OpenAI는 .message.content에 결과가 들어있음
            res_text = response.choices[0].message.content
            return json.loads(res_text)
            
        except Exception as e:
            print(f"LLM 처리 중 에러 발생: {e}")
            return {"filter_status": "F", "named_entities": [], "triples": []}

    def process_json_file(self, input_path, output_path):
        """
        Airflow 확장성 모델: 
        1. 기존 결과(output_path) 로드 및 중복 ID 추출
        2. 신규 데이터 중 미처리 건만 필터링
        3. T(True)로 판명된 데이터만 최종 보관
        """
        # 1. 기존에 이미 저장된 '진짜(T)' 뉴스 로드 (State Check)
        final_results = []
        processed_ids = set()
        
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                try:
                    final_results = json.load(f)
                    # 이미 저장된 기사들의 ID를 세트에 담아 광속 비교 준비
                    processed_ids = {art['id'] for art in final_results}
                except json.JSONDecodeError:
                    final_results = []

        # 2. 새로 수집된 뉴스(input_path) 로드
        with open(input_path, 'r', encoding='utf-8') as f:
            new_articles = json.load(f)

        print(f"🔄 분석 시작: 신규 수집 {len(new_articles)}건 (기존 DB 내 {len(processed_ids)}건 제외)")

        newly_added_count = 0
        
        for art in new_articles:
            # 중복 제거: 이미 최종 파일에 있는 ID라면 무조건 패스 (전역 중복 제거)
            if art['id'] in processed_ids:
                continue
                
            # 1차 휴리스틱 필터
            score = self.calculate_heuristic_score(art)
            
            if score >= 3:
                print(f"🔍 [Pass Filter] 점수 {score}점: {art['title'][:30]}...")
                
                # 2차 LLM 검증 및 추출
                llm_data = self.call_llm_extractor(art)
                
                # 핵심 로직: LLM 결과가 'T'인 경우에만 최종 결과물에 추가
                if llm_data.get('filter_status') == 'T':
                    art.update(llm_data)
                    final_results.append(art)
                    processed_ids.add(art['id']) # 이번 배치 내 중복 방지
                    newly_added_count += 1
                    print(f"✅ [Final T] 유효 뉴스 추가 완료!")
                    time.sleep(0.5) 
                else:
                    print(f"❌ [LLM F] 관련 없음 판정")
            else:
                # 점수 미달은 기록조차 하지 않음 (용량 절약)
                continue

        # 3. 최종적으로 'T'인 데이터들만 모아서 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)
        
        print(f"\n✨ 완료! 새로운 유효 뉴스 {newly_added_count}건이 금고({output_path})에 업데이트되었습니다.")