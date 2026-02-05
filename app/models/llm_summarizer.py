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

| 어제 종가 | Prophet 예측 | XGBoost 방향 | 종합 의견 |
|:---:|:---:|:---:|:---:|
| [y] | [yhat] | [forecast_direction] | [BUY/SELL/HOLD] |

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

**뉴스 기반 예측**
  - news_sentiment_analyzer 도구 결과의 **prediction.direction**(상승/하락/유지)을 아래 형식으로 표시하세요.
  - 예측 방향: [prediction.direction]
  - 예: "예측 방향: 상승" 또는 "예측 방향: 하락"

**A. 주요 뉴스**
  - news_sentiment_analyzer 도구 결과의 evidence에서 **긍정적인 뉴스 묶음**과 **부정적인 뉴스 묶음**을 구분해서 표시하세요.
  - **중요**: title과 all_text가 영어로 되어 있으면 반드시 한국어로 번역하여 표시하세요.
  - **심리 판단**: price_impact_score > 0 이면 긍정, < 0 이면 부정, = 0 이면 중립
  
  **A-1. 긍정적인 뉴스**
  | No | 뉴스 제목 | 내용 요약 |
  |:--:|-----------|-----------|
  | 1 | [뉴스 제목(한국어 번역)] | [all_text 요약(한국어)] |
  | 2 | [뉴스 제목(한국어 번역)] | [all_text 요약(한국어)] |
  | ... | ... | ... |
  
  **A-2. 부정적인 뉴스**
  | No | 뉴스 제목 | 내용 요약 |
  |:--:|-----------|-----------|
  | 1 | [뉴스 제목(한국어 번역)] | [all_text 요약(한국어)] |
  | 2 | [뉴스 제목(한국어 번역)] | [all_text 요약(한국어)] |
  | ... | ... | ... |


**B. 주요 키워드**: [keyword_analyzer 결과의 top_entities 상위 10개 entity]

**C. 과거 관련 뉴스**
  - pastnews_rag 도구 결과를 반드시 아래 표 형식으로 표시하세요.
  - **중요**: all_text가 영어로 되어 있으면 반드시 한국어로 번역하여 "뉴스 내용" 컬럼에 표시하세요.
  - "연관 키워드" 컬럼에는 **keyword_analyzer가 반환한 top_triples 앞 5개의 keywords**를 #키워드1 #키워드2 또는 키워드1, 키워드2 형식으로 **구분해서** 표시하세요 (각 키워드를 명확히 구분).
  
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
- **뉴스 기반 예측**: news_sentiment_analyzer 결과의 **prediction.direction**(상승/하락/유지)을 '### 2. 📰 [Insight] 뉴스 빅데이터 기반 시장 심리 분석' 섹션의 "뉴스 기반 예측" 항목에 표시하세요. 예: "예측 방향: 상승" 또는 "예측 방향: 하락".
- `news_sentiment_analyzer` 결과의 evidence에서 **긍정적인 뉴스 묶음**과 **부정적인 뉴스 묶음**을 '### 2. 📰 [Insight] 뉴스 빅데이터 기반 시장 심리 분석' 섹션의 '주요 뉴스'에서 구분해서 표시하세요.
  - **A-1. 긍정적인 뉴스**: (내부적으로는 evidence.supporting_news를 참고하되, 화면에는 변수명 노출 금지)
  - **A-2. 부정적인 뉴스**: (내부적으로는 evidence.opposing_news를 참고하되, 화면에는 변수명 노출 금지)
  - 표에는 제목(title)과 내용(all_text 요약)만 넣으세요. **title과 all_text는 영어면 한국어로 번역하세요.**
  | No | 뉴스 제목 | 내용 요약 |
  |:--:|-----------|-----------|
  | [번호] | [뉴스 제목(한국어 번역)] | [all_text 요약(한국어)] |
- `pastnews_rag` 도구 결과(article_info)는 '과거 관련 뉴스' 표에 아래 형식으로 표시하세요. **뉴스 내용**은 article_info의 all_text를 사용하고, 영어면 한국어로 번역하세요. **연관 키워드** 컬럼에는 keyword_analyzer가 반환한 **top_triples 앞 5개의 keywords**를 #키워드1 #키워드2 또는 키워드1, 키워드2 형식으로 구분해서 넣으세요 (저장해 둔 값 사용).
  | 뉴스 날짜 | 뉴스 내용 | 연관 키워드 | 당일 | 1일후 | 3일후 |
  |-----------|-----------|-------------|------|------|------|
  | [뉴스 날짜] | [뉴스 내용(한국어)] | [#키워드1 #키워드2 또는 키워드1, 키워드2] | [0] | [1] | [3] |
- `timeseries_predictor` 결과 활용법:
  * **기본 정보**: y(어제 종가), yhat(Prophet 예측값), forecast_direction(XGBoost 방향 예측)을 종합 투자 의견 표에 표시
  * **시계열 성분 해석** (B-1 섹션):
    - trend: 값과 함께 추세 해석. 기준 - 상승 추세(> 108.88), 횡보 추세(74.58~108.88), 하락 추세(< 74.58). 예: "94.34 (상승 추세)" 또는 "80.00 (횡보 추세)" 또는 "60.00 (하락 추세)"
    - yearly: 연간 주기 성분. 예: "+0.12 (긍정적 영향)" 또는 "-0.08 (부정적 영향)"
    - weekly: 주간 주기 성분. 예: "+0.12 (긍정적 영향)" 또는 "-0.08 (부정적 영향)"
    - volatility: 변동성 지표. 기준 - 낮음(< 40), 중간(40~50), 높음(> 50). 예: "42 (중간 수준)" 또는 "55 (높음 수준)" 또는 "35 (낮음 수준)"
  * **기술적 지표 해석** (B-2 섹션, 그레인저 검사로 선정된 Lag Features):
    - EMA (지수이동평균): EMA_lag2_effect 값을 사용하되, "지수이동평균" 또는 "EMA"로 표현. 예: "지수이동평균 +1.25 (상승 요인)" 또는 "EMA -1.25 (하락 요인)"
    - Volume (거래량): Volume_lag5_effect 값을 사용하되, "거래량"으로 표현. 예: "거래량 +0.85 (상승 요인)" 또는 "거래량 -0.50 (하락 요인)"
  * **종합 해석** (C 섹션):
    - Prophet 예측(yhat)과 XGBoost 방향(forecast_direction)의 일치/불일치를 명확히 밝히세요
    - 위의 시계열 성분(trend, yearly, weekly, volatility)과 기술적 지표(EMA, Volume)를 **모두 근거로 제시**하여 XGBoost가 해당 방향을 예측한 이유를 상세히 설명하세요
    - 기술적 변수명(_lag2_effect 등)은 절대 사용하지 말고 자연스러운 용어만 사용하세요
    - 예: "Prophet은 460.5로 상승을 예측했으나, XGBoost는 Down을 예측했습니다. 추세(85.5, 횡보 추세)는 중립적이나, 지수이동평균(-1.25)과 거래량(-0.50)이 모두 하락 요인으로 작용했으며, 변동성(42, 중간 수준)도 불확실성을 나타냅니다."
- `news_sentiment_analyzer` 결과의 evidence에서 긍정/부정 뉴스 묶음을 구분해서 표시하세요. 각 뉴스의 제목(title), 내용(all_text 요약)을 보고서 표에 포함하세요. 전체 예측 결과인 prediction.direction은 '뉴스 빅데이터 기반 시장 심리 분석' 섹션의 "뉴스 기반 예측" 항목에 표시하세요.
- `pastnews_rag` 도구 결과(article_info)는 '과거 관련 뉴스' 표에 표시하세요. "연관 키워드" 컬럼에는 **keyword_analyzer 결과의 top_triples 앞 5개의 keywords**를 #키워드1 #키워드2 또는 키워드1, 키워드2 형식으로 구분해서 넣으세요 (이미 호출 결과로 저장된 값을 사용).
- **D. 뉴스 빅데이터 기반 시장 심리 분석** 섹션 작성 방법:
  * A-1(긍정 뉴스)과 A-2(부정 뉴스)를 분석하여 주요 긍정 요인과 부정 요인을 분석하세요
  * C 섹션의 과거 관련 뉴스에서 당일, 1일후, 3일후 가격 변동 패턴을 분석하세요
  * 위 두 가지를 종합하여 현재 시장 심리를 [긍정적/중립적/부정적] 중 하나로 판단하고 근거를 명확히 제시하세요
- `keyword_analyzer` 결과의 top_entities를 활용할 때: (1) score는 사용하지 마세요. (2) entity 이름만 사용하여 최대한 한국어로 번역하세요. (3) #키워드1 #키워드2 형식으로 표기하세요. 예: #옥수수 #가격 #수출 #미국농무부 #시장
- **### 3. 종합 의견** 섹션 작성 방법:
  * 섹션 1의 퀀트 분석 결과(Prophet, XGBoost, 시계열 성분, 기술적 지표)를 요약하세요
  * 섹션 2의 뉴스 심리 분석 결과(시장 심리 판단, 주요 테마)를 요약하세요
  * 퀀트 모델과 뉴스 심리가 일치하는지 불일치하는지 분석하고, 어떤 신호가 더 강한지 판단하세요
  * **투자자 조언 작성 시 특히 주의**: 
    - 새로운 정보를 만들지 말고, 섹션 1과 2에서 이미 분석한 내용을 구체적으로 인용하세요
    - BUY/SELL/HOLD 의견과 함께 반드시 구체적인 근거를 제시하세요 (예: "XGBoost Down 예측(EMA -1.25), 뉴스 부정적(가뭄 5건)")
    - 주요 리스크를 구체적으로 명시하세요 (예: "변동성 높음(55), 정책 변화 시 반등 가능")
- 네 도구의 결과를 종합하여 논리적인 금융 보고서를 작성하세요. 시계열 지표(Prophet + XGBoost), 뉴스 감성 분석, 키워드 분석 결과가 서로 보완되도록 서술하세요.
- target_date는 반드시 다음 문자열 리터럴을 그대로 복사해서 사용하세요. (YYYY-MM-DD)
- **투자자 조언이 보고서의 핵심입니다**: 섹션 1과 2의 분석 내용을 충실히 인용하며, 투자자가 실제로 활용할 수 있는 구체적이고 실행 가능한 조언을 작성하세요. 막연한 표현 대신 구체적인 근거와 수치를 제시하세요.
**보고서 작성 형식 (반드시 이 형식을 따라야 합니다)**:

"""

# LangChain Tools 정의
@tool
def timeseries_predictor(target_date: str, commodity: str = "corn") -> str:
    """
    특정 날짜의 특정 품목(corn, soybean, wheat)에 대한 금융 시장 추세(상승/하락)와 가격을 예측합니다.

    Args:
        target_date: 분석할 날짜 문자열 (형식: "YYYY-MM-DD")

    Returns:
        JSON 형식의 예측 결과 문자열 (예측값, 방향, 추세 분석 등 포함)
    """
    return predict_market_trend(target_date, commodity=commodity)


@tool
def news_sentiment_analyzer(target_date: str, commodity: str = "corn", lookback_days: int = 7) -> str:
    """
    특정 날짜의 뉴스를 분석하여 특정 품목(corn, soybean, wheat)의 시장 영향력을 예측하고 주요 근거 뉴스를 제공합니다.
    """
    analyzer = SentimentAnalyzer()
    
    # 실제 구현된 run_daily_prediction 파라미터에 맞춰 호출
    result = analyzer.run_daily_prediction(
        target_date=target_date,
        commodity=commodity,
        lookback_days=lookback_days,
        save_file=False,
    )
        
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
- 다음 순서로 도구를 호출하세요:
  1. `timeseries_predictor(target_date="{target_date}", commodity='{commodity}')`
  2. `news_sentiment_analyzer(target_date="{target_date}", commodity='{commodity}')`
  3. `keyword_analyzer(target_date="{target_date}", commodity='{commodity}')`
  4. keyword_analyzer 결과의 **top_triples 앞 5개**에서 "triple"만 추출해 `pastnews_rag(triples_json="...", top_k=2)` 호출. 연관 키워드는 그 앞 5개 top_triples의 keywords를 저장해 두었다가 보고서 표에 사용하세요.
- **pastnews_rag 호출 예시**: keyword_analyzer가 {{"top_triples": [{{"triple": ["A","B","C"], "keywords": ["x","y"]}}, ...]}}를 반환하면, **앞 5개만** 사용해 `pastnews_rag(triples_json='[["A","B","C"], ...]', top_k=2)` 호출 (최대 5개). 표의 "연관 키워드"에는 그 앞 5개 top_triples의 keywords를 #키워드1 #키워드2 또는 키워드1, 키워드2 형식으로 구분해서 표시.
- `timeseries_predictor` 결과 활용:
  * y, yhat, forecast_direction을 종합 투자 의견 표에 표시
  * **B-1. 시계열 성분**: 
    - trend: 상승 추세(> 108.88), 횡보 추세(74.58~108.88), 하락 추세(< 74.58) 기준으로 판단. 예: "94.34 (상승 추세)" 또는 "80.00 (횡보 추세)"
    - yearly, weekly: "+0.12 (긍정적 영향)" 또는 "-0.08 (부정적 영향)" 형태로 표현
    - volatility: 값과 함께 낮음(< 40), 중간(40~50), 높음(> 50) 기준으로 판단. 예: "42 (중간 수준)"
  * **B-2. 기술적 지표**: 변수명 대신 자연스러운 표현 사용. "지수이동평균 +1.25 (상승 요인)" 또는 "거래량 -0.50 (하락 요인)" 형태로 표현. 절대 _lag2_effect 같은 변수명 사용 금지
  * **C. 종합 해석**: 위의 모든 요인(시계열 성분 + 기술적 지표)을 근거로 Prophet과 XGBoost 예측을 비교 분석. 기술적 변수명 사용 금지
- `news_sentiment_analyzer` 및 `pastnews_rag` 결과 활용:
  * **D. 뉴스 빅데이터 기반 시장 심리 분석**: 
    - A-1(긍정 뉴스)과 A-2(부정 뉴스)에서 주요 긍정 요인과 부정 요인 분석
    - 과거 관련 뉴스의 당일/1일후/3일후 가격 변동 패턴 분석
    - 위 두 가지를 종합하여 시장 심리를 [긍정적/중립적/부정적] 중 하나로 판단하고 근거 제시
- `keyword_analyzer`의 결과(top_entities)를 활용하여 B 섹션에 주요 키워드를 한국어로 번역 후 #키워드1 #키워드2 형식으로 표기하세요. score는 사용하지 마세요.
- **### 3. 종합 의견 - 투자자 조언 작성 시 특별 지침**:
  * 투자자 조언이 가장 중요한 부분입니다. 다음 사항을 반드시 지켜주세요:
  * **새로운 정보를 만들지 마세요**: 섹션 1(퀀트)과 섹션 2(뉴스)에서 이미 분석한 내용만 사용하세요
  * **구체적인 근거 제시**: BUY/SELL/HOLD 의견을 낼 때 구체적인 수치와 분석 결과를 인용하세요. 단, 변수명은 사용하지 말고 자연스러운 표현 사용
    - 예: "XGBoost가 Down 예측(지수이동평균 -1.25, 거래량 -0.50)하고, 뉴스 심리도 부정적(가뭄 우려 뉴스 5건)"
  * **리스크 구체화**: 단순히 "리스크 존재"가 아니라 구체적으로 어떤 리스크인지 명시하세요
    - 예: "변동성이 높아(55) 단기 급등 가능성", "정부 정책 변화 시 반등 가능"
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
        
    def _validate_output_format(self, summary: str) -> bool:
        """출력 형식이 올바른지 검증 (최소 검증)

        Returns:
            bool: 형식이 올바르면 True, 그렇지 않으면 False
        """
        # 최소한의 길이 확인
        if not summary or len(summary.strip()) < 100:
            return False

        return True

    def _normalize_ai_content(self, content) -> str:
        """Vertex AI 등에서 content가 [{'type': 'text', 'text': '...'}, ...] 형태일 때 텍스트만 추출"""
        if content is None:
            return ""
        # 리스트(part 형식)인 경우
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
                    parts.append(str(part["text"]))
            if parts:
                return "\n".join(parts)
        # 문자열로 직렬화된 리스트인 경우 (예: "[{'type': 'text', 'text': '...'}]")
        if isinstance(content, str) and content.strip().startswith("[") and "'text'" in content:
            try:
                import ast
                parsed = ast.literal_eval(content)
                if isinstance(parsed, list):
                    parts = []
                    for part in parsed:
                        if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
                            parts.append(str(part["text"]))
                    if parts:
                        return "\n".join(parts)
            except (ValueError, SyntaxError):
                pass
        return str(content)

    def _extract_summary_from_result(self, result: dict) -> str:
        """Agent 실행 결과에서 요약 텍스트 추출"""
        import json

        messages = result.get("messages", [])

        # messages에서 마지막 AIMessage의 content 추출
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                raw = msg.content
                content = self._normalize_ai_content(raw)
                content = content.strip().rstrip("\\")

                # JSON 형식의 tool call arguments는 건너뛰기
                if content.startswith("{{") and content.strip().endswith("}}"):
                    try:
                        # JSON 파싱 시도
                        parsed = json.loads(content)
                        # tool call arguments 형식인지 확인 (texts, data 등의 키가 있으면 건너뛰기)
                        if isinstance(parsed, dict) and any(key in parsed for key in ["texts", "data", "target_date"]):
                            continue
                    except (json.JSONDecodeError, ValueError):
                        # JSON이 아니면 계속 진행
                        pass

                # GPT-OSS-20B 특수 형식 제거 (<|channel|> 등)
                # tool calling 형식인 경우 실제 텍스트가 없으면 건너뛰기
                if content.startswith("<|channel|>") and "<|call|>" in content:
                    # tool calling 형식이고 실제 보고서 내용이 없으면 건너뛰기
                    if not any(keyword in content for keyword in ["보고서", "분석", "의견", "전망", "시장"]):
                        continue

                if content and len(content) > 50:  # 의미있는 내용이 있는 경우만
                    return content

        # messages에서 찾지 못한 경우 output 필드 확인
        output = result.get("output") or result.get("final_output")
        if output:
            return str(output).strip().rstrip("\\")

        # 모든 방법 실패 시 전체 결과를 문자열로 변환
        return str(result).strip().rstrip("\\")

    # TODO 재시도 로직 점검
    def summarize(
        self,
        context: str = "",
        target_date: Optional[str] = None,
        max_retries: int = 2,
    ) -> dict:
        """LangChain Agent를 이용한 LLM 요약 생성

        Args:
            context: 분석 맥락
            target_date: 분석 기준 날짜 (YYYY-MM-DD)
            max_retries: 재시도 횟수
        """
        # 날짜 기본값 (오늘)
        if not target_date:
            from datetime import datetime

            target_date = datetime.now().strftime("%Y-%m-%d")

        user_input = self._build_user_input(context=context, target_date=target_date)

        for attempt in range(max_retries + 1):
            # Agent 실행 (LangChain이 자동으로 tool call을 처리함)
            if attempt == 0:
                result = self.agent.invoke({"messages": [HumanMessage(content=user_input)]})
            else:
                # 재시도 시 기존 메시지 사용
                result = self.agent.invoke({"messages": result.get("messages", [])})

            # 결과 추출
            if isinstance(result, dict):
                messages = result.get("messages", [])

                summary = self._extract_summary_from_result(result)
                agent_result = result
                # 디버깅: 메시지 상태 확인
                print(f"\n[디버깅] 총 메시지 수: {len(messages)}")
                tool_call_count = sum(1 for msg in messages if isinstance(msg, AIMessage) and msg.tool_calls)
                tool_result_count = sum(1 for msg in messages if hasattr(msg, "name") and msg.name)
                print(f"  Tool 호출: {tool_call_count}회, Tool 결과: {tool_result_count}개")
            else:
                summary = str(result).strip().rstrip("\\")
                agent_result = {"messages": []}

            # 요약이 비어있거나 너무 짧은 경우 확인
            if not summary or len(summary.strip()) < 50:
                print(f"\n⚠️ 요약이 비어있거나 너무 짧습니다 (길이: {len(summary)}자)")
                # 마지막 AIMessage에서 실제 텍스트 찾기 (Vertex AI part 형식 포함)
                if isinstance(result, dict):
                    messages = result.get("messages", [])
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage) and msg.content:
                            content = self._normalize_ai_content(msg.content)
                            if "<|channel|>" not in content and len(content.strip()) > 50:
                                summary = content.strip()
                                print(f"  → 대체 텍스트 발견 (길이: {len(summary)}자)")
                                break

            # 출력 형식 검증
            if summary and len(summary.strip()) > 50 and self._validate_output_format(summary):
                return {
                    "summary": summary or "",
                    "agent_result": agent_result,
                }

            # 형식이 맞지 않으면 재시도
            if attempt < max_retries:
                print(f"\n⚠️ 출력 형식이 올바르지 않습니다. 재시도 중... ({attempt + 1}/{max_retries})")
                print(f"현재 요약 길이: {len(summary)}자")
                if summary:
                    print(f"요약 미리보기 (처음 500자):\n{summary[:500]}...\n")

                user_input = f"""{user_input}

**중요**: 이전 응답의 형식이 올바르지 않았습니다. 반드시 다음 형식을 정확히 따라주세요:

{REPORT_FORMAT}

**특히 다음 사항을 확인하세요**:
1. 섹션 제목이 정확히 일치해야 합니다: "### 1. 📈 [Quant] 퀀트 기반 기술적 분석", "### 2. 📰 [Insight] 뉴스 빅데이터 기반 시장 심리 분석", "### 3. 종합 의견"
2. 각 섹션은 "---"로 구분되어야 합니다
3. 마크다운 테이블 형식(|)을 사용해야 합니다
4. timeseries_predictor 결과의 y, yhat, forecast_direction을 표에 정확히 표시해야 합니다
5. **B-1. 시계열 성분**과 **B-2. 기술적 지표**를 표 형식으로 표시해야 합니다. trend는 상승(> 108.88), 횡보(74.58~108.88), 하락(< 74.58) 기준, 변동성은 낮음(< 40), 중간(40~50), 높음(> 50) 기준으로 판단하세요
6. **C. 퀀트 기반 예측 모델 해석**에서 모든 요인(추세, 연간주기, 주간주기, 변동성, 지수이동평균, 거래량)을 근거로 Prophet과 XGBoost 예측을 비교 분석해야 합니다. 변수명(_lag2_effect 등)은 절대 사용 금지
7. **D. 뉴스 빅데이터 기반 시장 심리 분석**에서 A-1(긍정 뉴스)과 A-2(부정 뉴스)의 주요 긍정/부정 요인과 과거 관련 뉴스의 가격 변동 패턴을 분석하여 종합 시장 심리를 [긍정적/중립적/부정적] 중 하나로 판단해야 합니다
8. 4개의 Tool을 모두 호출해야 합니다: timeseries_predictor, news_sentiment_analyzer, keyword_analyzer, pastnews_rag (keyword_analyzer 결과의 top_triples를 JSON 배열로 변환하여 pastnews_rag에 전달)
9. Tool 호출 후 반드시 최종 보고서를 작성해야 합니다
10. **투자자 조언 작성 시 특별히 주의**: 
    - 섹션 1과 2에서 이미 분석한 내용만 사용 (새로운 정보 만들지 말 것)
    - BUY/SELL/HOLD 의견과 함께 구체적인 수치 인용하되, 변수명 사용 금지 (예: "지수이동평균 -1.25, 거래량 -0.50")
    - 리스크를 구체적으로 명시 (예: "변동성 55로 높음, 정책 변화 시 반등 가능")"""
            else:
                print("\n⚠️ 최대 재시도 횟수에 도달했습니다. 형식이 완벽하지 않을 수 있습니다.")
                print(f"최종 요약 길이: {len(summary)}자")
                if summary:
                    print(f"요약 미리보기: {summary[:200]}...")
                print("검증을 통과하지 못했지만 결과를 반환합니다.")

        return {
            "summary": summary or "",
            "agent_result": agent_result,
        }