from typing import Optional, Tuple, Union, Dict, Any, List
from langchain_core.tools import tool
import subprocess
import json
import os
import sys
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage, AIMessage

from langchain.agents import create_agent
from langchain_google_vertexai import ChatVertexAI

from config.settings import (
    GENERATE_MODEL_NAME,
    GENERATE_MODEL_TEMPERATURE,
    GENERATE_MODEL_MAX_TOKENS,
    VERTEX_AI_PROJECT_ID,
    VERTEX_AI_LOCATION,
)
from models.timeseries_predictor import predict_market_trend
from models.sentiment_analyzer import SentimentAnalyzer
from models.keyword_analyzer import analyze_keywords as _analyze_keywords
from models.pastnews_rag_runner import run_pastnews_rag as _run_pastnews_rag


REPORT_FORMAT = """
**날짜**: (YYYY-MM-DD) | **종목**: [분석 대상 품목명] 

| 어제 종가 | Prophet 예측 | XGBoost 방향 | 뉴스 심리 | 종합 의견 |
|:---:|:---:|:---:|:---:|:---:|
| [y] | [yhat] | [forecast_direction] | [긍정/부정/중립] | [BUY/SELL/HOLD] |

---

### 1. 📈 [Quant] 퀀트 기반 기술적 분석

**A. 가격 예측**
* **어제 종가**: [y]
* **Prophet 예측값**: [yhat] 
* **XGBoost 방향 예측**: [forecast_direction] (Up/Down)

**B. 주요 변동 요인**

**B-1. 시계열 성분**

| 지표 | 값 | 해석 | 설명 |
|------|-----|------|------|
| 추세 (trend) | [값] | [상승/횡보/하락] 추세 | 추세 지표|
| 연간 주기 (yearly) | [값] | [긍정적/부정적/중립] 영향 | 계절적 요인으로 인한 연간 패턴 |
| 주간 주기 (weekly) | [값] | [긍정적/부정적/중립] 영향 | 요일별 패턴 |
| 변동성 (volatility) | [값] | [높음/중간/낮음] 수준 | 시장 불확실성 지표 |

**B-2. 기술적 지표**

| 지표 | 값 | 해석 | 설명 |
|------|-----|------|------|
| EMA (지수이동평균) | [값] | [상승/하락] 요인 | EMA 영향 지표 |
| Volume (거래량) | [값] | [상승/하락] 요인 | 거래량 영향 지표 |

**C. 퀀트 기반 예측 모델 해석**

* **Prophet vs XGBoost 비교**:
  - Prophet 예측: [yhat] ([상승/하락])
  - XGBoost 예측: [forecast_direction] (Up/Down)
  - 일치 여부: [일치/불일치]

* **핵심 근거 분석**:
  - **시계열 성분**: 추세([trend], [상승/횡보/하락] 추세), 연간주기([yearly]), 주간주기([weekly]), 변동성([volatility], [높음/중간/낮음] 수준)을 종합하면 [분석 내용]
  - **기술적 지표**: 지수이동평균(EMA_lag2_effect: [값])과 거래량(Volume_lag5_effect: [값])은 [상승/하락] 요인으로 작용
  - **종합 판단**: Prophet이 [yhat]로 예측하고 XGBoost가 [forecast_direction]을 예측한 이유를 위 요인들을 바탕으로 서술. 단, 변수명(EMA_lag2_effect 등)은 사용하지 말고 "지수이동평균", "거래량" 등 자연스러운 표현 사용
---
### 2. 📰 [Insight] 뉴스 빅데이터 기반 시장 심리 분석

**A. 주요 뉴스 (evidence_news)**
  - news_sentiment_analyzer 도구 결과를 반드시 아래 표 형식으로 표시하세요.
  - **중요**: title과 all_text가 영어로 되어 있으면 반드시 한국어로 번역하여 표시하세요.
  - **심리 판단**: price_impact_score > 0 이면 긍정, < 0 이면 부정, = 0 이면 중립
  
  | No | 뉴스 제목 | 내용 요약 | 심리 |
  |:--:|-----------|-----------|:--------:|
  | 1 | [뉴스 제목(한국어 번역)] | [all_text 요약(한국어)] | [긍정/부정/중립] |
  | 2 | [뉴스 제목(한국어 번역)] | [all_text 요약(한국어)] | [긍정/부정/중립] |
  | ... | ... | ... | ... |


**B. 주요 키워드**: [keyword_analyzer 결과의 top_entities 상위 10개 entity]

**C. 과거 관련 뉴스**
  - pastnews_rag 도구 결과를 반드시 아래 표 형식으로 표시하세요.
  - **중요**: description이 영어로 되어 있으면 반드시 한국어로 번역하여 "뉴스 제목" 컬럼에 표시하세요.
  
  | 뉴스 날짜 | 뉴스 내용 | 당일 | 1일후 | 3일후 |
  |-----------|-----------|------|------|------|
  | [뉴스 날짜] | [뉴스 내용(한국어 번역)] | [0] | [1] | [3] |
  | [뉴스 날짜] | [뉴스 내용(한국어 번역)] | [0] | [1] | [3] |
  | ... | ... | ... | ... | ... |

**D. 뉴스 빅데이터 기반 시장 심리 분석**

  * **주요 뉴스 분석**
    - 주요 긍정 요인: [긍정적 뉴스들의 공통 주제/키워드]
    - 주요 부정 요인: [부정적 뉴스들의 공통 주제/키워드]

  * **과거 유사 상황 분석**
    - C 섹션의 과거 관련 뉴스를 분석하여 당시 시장 반응(당일, 1일후, 3일후 가격 변동)을 서술
    - 공통 패턴: [과거 유사 뉴스 발생 시 가격 변동 패턴]

  * **종합 시장 심리**
    - 판단: [긍정적/중립적/부정적]
    - 근거: [위의 분석을 바탕으로 한 종합 판단 이유]
---

### 3. 종합 의견

* **퀀트 분석 요약** :
  - Prophet 예측: [yhat] ([상승/하락])
  - XGBoost 방향: [forecast_direction] (Up/Down)
  - 주요 근거: [trend, EMA, Volume 등 핵심 요인 요약]

* **뉴스 심리 분석 요약** :
  - 시장 심리: [긍정적/중립적/부정적]
  - 주요 테마: [핵심 키워드 및 테마]

* **최종 투자 의견**:
  - **단기 전망** : [퀀트 + 뉴스 분석 종합]
  - **핵심 근거**: [퀀트 모델과 뉴스 심리가 일치/불일치하는지, 어떤 신호가 더 강한지]
  - **투자자 조언**: 
    * **투자 의견**: [BUY/SELL/HOLD]
    * **의견 근거**: [섹션 1의 퀀트 분석과 섹션 2의 뉴스 심리 분석을 구체적으로 인용하며 종합. 예: "XGBoost가 Down을 예측했고(EMA -1.25, Volume -0.50), 뉴스 심리도 부정적(가뭄 우려 5건)이므로 SELL"]
    * **주요 리스크**: [예상되는 리스크 요인을 구체적으로 명시. 예: "변동성이 높아(55) 단기 급등 가능성 존재", "정부 정책 변화 시 반등 가능"]

**중요**: 
- 반드시 위 형식을 정확히 따라야 합니다.
- 표 형식은 마크다운 테이블 문법을 사용하세요.
- 섹션 번호와 제목은 정확히 일치해야 합니다.
- 각 섹션은 "---"로 구분하세요.
- 주요 키워드는 #키워드1 #키워드2 형식으로 표기
- 뉴스 관련 내용이 영어로 되어 있으면 반드시 한국어로 번역하여 표시하세요. 원문을 그대로 표시하지 마세요.
- 언어는 반드시 순수 한국어(한글)만 사용하세요."""

SYSTEM_PROMPT = """당신은 전문 금융 분석가입니다.

**사용 가능한 도구**:
1. timeseries_predictor: Prophet + XGBoost 하이브리드 시계열 예측
   - target_date: 분석할 대상 날짜 (형식: "YYYY-MM-DD")
   - commodity: 상품명 (corn, soybean, wheat)
   - 설명: 특정 품목의 가격 예측(yhat)과 방향 예측(forecast_direction)을 반환합니다.

2. news_sentiment_analyzer: 뉴스 기반 시장 영향력 분석 및 근거 추출
   - target_date: 분석할 대상 날짜 (형식: "YYYY-MM-DD")
   - commodity: 상품명 (corn, soybean, wheat)
   - lookback_days: 조회할 과거 일수 (기본 7일)
   - 설명: 해당 날짜 전후의 뉴스를 분석하여 시장 상승/하락 확률을 예측하고, 주요 근거 뉴스를 반환합니다.

3. keyword_analyzer: 뉴스 기사의 주요 키워드 분석
   - target_date: 분석할 대상 날짜 (형식: "YYYY-MM-DD")
   - commodity: 상품명 (corn, soybean, wheat)
   - days: 분석할 일수 (기본 3일)
   - 설명: 뉴스 기사에서 핵심 키워드와 Triple(S-V-O) 관계를 추출합니다.

4. pastnews_rag: 전달받은 triples로 유사 뉴스 및 과거 가격 조회
   - triples_json: keyword_analyzer 결과의 top_triples 배열을 JSON으로 전달
   - commodity: 상품명 (corn, soybean, wheat)
   - top_k: 유사 결과 개수 (기본 2~5)
   - 설명: 현재의 주요 뉴스 상황이 과거 언제 발생했는지 찾고, 당시의 가격 변동을 보여줍니다.

**도구 사용 규칙**:
- 모든 도구 호출 시 현재 분석 중인 `commodity`를 명시적으로 전달하세요.
- keyword_analyzer 호출 후, 결과의 top_triples **앞 5개**를 추출하여 pastnews_rag에 전달하세요.
- 이전 도구가 오류를 반환하더라도, 네 도구를 반드시 모두 호출한 뒤에만 보고서를 작성하세요.
- 모든 영어 텍스트(뉴스 제목, 내용 등)는 반드시 한국어로 번역하여 보고서에 포함하세요.
"""

# LangChain Tools 정의
@tool
def timeseries_predictor(target_date: str, commodity: str = "corn") -> str:
    """
    특정 날짜의 특정 품목(corn, soybean, wheat)에 대한 금융 시장 추세와 가격을 예측합니다.
    """
    return predict_market_trend(target_date, commodity=commodity)


@tool
def news_sentiment_analyzer(target_date: str, commodity: str = "corn", lookback_days: int = 7) -> str:
    """
    특정 날짜의 뉴스를 분석하여 특정 품목(corn, soybean, wheat)의 시장 영향력을 예측하고 주요 근거 뉴스를 제공합니다.
    """
    analyzer = SentimentAnalyzer()
    # 팀원들이 추가한 run_daily_prediction 메서드 사용 (commodity 인자 전달 가능하다고 가정)
    try:
        result = analyzer.run_daily_prediction(
            target_date=target_date,
            lookback_days=lookback_days,
            commodity=commodity,
            filter_status="T",
            save_file=False,
        )
    except TypeError:
        # 만약 run_daily_prediction이 아직 commodity를 안 받는다면 기존 메서드로 폴백
        result = analyzer.predict_market_impact(target_date, commodity=commodity)
        
    return json.dumps(result, ensure_ascii=False)


@tool
def keyword_analyzer(target_date: str, commodity: str = "corn", days: int = 3) -> str:
    """
    특정 날짜 기준으로 뉴스 기사의 주요 키워드를 분석합니다. (품목별 필터링 지원)
    """
    print(f"[keyword_analyzer] 실행 시작 (commodity: {commodity})", flush=True)
    result = json.loads(_analyze_keywords(target_date=target_date, commodity=commodity, days=days, top_k=10))
    top_entities = result.get("top_entities", [])[:10]
    top_triples = result.get("top_triples", [])
    print("[keyword_analyzer] 종료", flush=True)
    return json.dumps({"top_entities": top_entities, "top_triples": top_triples}, ensure_ascii=False, indent=2)


@tool
def pastnews_rag(triples_json: str, commodity: str = "corn", top_k: int = 5) -> str:
    """
    전달받은 triples로 특정 품목(corn, soybean, wheat)의 유사 뉴스를 검색하고 과거 가격 정보를 조회합니다.
    """
    print(f"[pastnews_rag] 실행 시작 (commodity: {commodity})", flush=True)
    triples = []
    if triples_json and triples_json.strip():
        try:
            parsed = json.loads(triples_json)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        triples.append(list(item[:3]))
                    elif isinstance(item, dict) and "triple" in item and isinstance(item["triple"], (list, tuple)) and len(item["triple"]) >= 3:
                        triples.append(list(item["triple"][:3]))
        except (json.JSONDecodeError, TypeError):
            pass
    
    # 앞 5개만 사용 (리소스 제한)
    triples = triples[:5] if triples else []
    result = _run_pastnews_rag(triples=triples if triples else None, commodity=commodity, top_k=top_k)
    print("[pastnews_rag] 종료", flush=True)
    return json.dumps(result, ensure_ascii=False, indent=2)


class LLMSummarizer:
    """Vertex AI를 사용하는 LangChain Agent를 이용한 통합 분석"""

    def __init__(self, model_name: str = None, project_id: str = None, location: str = None):
        self.model_name = model_name or GENERATE_MODEL_NAME
        self.project_id = project_id or VERTEX_AI_PROJECT_ID or self._get_project_id()
        self.location = location or VERTEX_AI_LOCATION
        self.llm = None
        self.agent = None
        self._initialize()

    def _get_project_id(self) -> str:
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            else:
                return VERTEX_AI_PROJECT_ID or "unknown"
        except Exception:
            return VERTEX_AI_PROJECT_ID or "unknown"

    def _create_llm(self) -> ChatVertexAI:
        return ChatVertexAI(
            model=self.model_name,
            project=self.project_id,
            location=self.location,
            temperature=GENERATE_MODEL_TEMPERATURE,
            max_output_tokens=GENERATE_MODEL_MAX_TOKENS,
        )

    def _initialize(self):
        self.llm = self._create_llm()
        tools = [timeseries_predictor, news_sentiment_analyzer, keyword_analyzer, pastnews_rag]
        llm_with_tools = self.llm.bind_tools(tools)
        self.agent = create_agent(model=llm_with_tools, tools=tools, system_prompt=SYSTEM_PROMPT)

    def _build_user_input(self, context: str, target_date: str, commodity: str) -> str:
        user_input = f"""다음 정보를 바탕으로 전문적인 금융 시장 분석 보고서를 작성해주세요.

**분석 대상 품목**: {commodity}
**분석 맥락**: {context or f"최근 {commodity} 시장 상황 분석"}
**분석 기준 일자**: {target_date}

- 모든 도구 호출 시 `commodity='{commodity}'` 인자를 반드시 전달하세요.
- `keyword_analyzer` 결과의 **top_triples 앞 5개**를 사용하여 `pastnews_rag`를 호출하세요.
- 모든 영어 텍스트는 한국어로 번역하여 보고서 서식을 엄격히 준수하여 작성하세요.

보고서 서식:
{REPORT_FORMAT}
"""
        return user_input

    def summarize(self, context: str = "", target_date: Optional[str] = None, commodity: str = "corn", max_retries: int = 2) -> dict:
        if not target_date:
            target_date = datetime.now().strftime("%Y-%m-%d")

        user_input = self._build_user_input(context=context, target_date=target_date, commodity=commodity)

        for attempt in range(max_retries + 1):
            try:
                result = self.agent.invoke({"messages": [HumanMessage(content=user_input)]})
                summary = self._extract_summary_from_result(result)
                
                if summary and len(summary.strip()) > 50:
                    return {"summary": summary, "agent_result": result}
            except Exception as e:
                print(f"⚠️ Agent 실행 중 오류 (시도 {attempt+1}): {e}")
                if attempt == max_retries: raise e

        return {"summary": "", "agent_result": {}}

    def _extract_summary_from_result(self, result: dict) -> str:
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if isinstance(msg.content, list):
                    return "\n".join([p["text"] for p in msg.content if "text" in p])
                return str(msg.content).strip()
        return ""