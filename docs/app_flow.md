# 질문 1.
User: runpipeline.py 함수 실행 흐름에 대해 상세히 정리하고 분석해서 알려줘. 이 엔트리 포인트가 결국 app directory에 있는 모든 파일들을 구동시킬거야. app 디렉토리의 프로젝트라고 봐도 무방해

# 답변 1.

이제 전체 파이프라인의 흐름을 파악했습니다. 아래에 상세한 분석 결과를 정리하겠습니다.

# 📊 run_pipeline.py 실행 흐름 상세 분석

## 1. 전체 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              run_pipeline.py (엔트리 포인트)                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         routes/orchestrator.py                                   │
│                      orchestrate_analysis() 함수 호출                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         models/llm_summarizer.py                                 │
│                        LLMSummarizer.summarize() 호출                            │
│                   (LangChain Agent + Vertex AI 기반)                             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                                       ▼
┌───────────────────────────────┐       ┌───────────────────────────────────────┐
│   @tool timeseries_predictor  │       │   @tool news_sentiment_analyzer       │
│  models/timeseries_predictor  │       │     models/sentiment_analyzer         │
└───────────────────────────────┘       └───────────────────────────────────────┘
          │                                               │
          ▼                                               ▼
┌───────────────────────────────┐       ┌───────────────────────────────────────┐
│  model/timeseries_model/      │       │  model/news_sentiment_model/          │
│     inference.py              │       │     inference_with_evidence.py        │
│  (XGBoost + Prophet 피처)      │       │  (XGBoost + FinBERT 감성)              │
└───────────────────────────────┘       └───────────────────────────────────────┘
          │                                               │
          └──────────────────────┬────────────────────────┘
                                 ▼
                    ┌────────────────────────┐
                    │  utils/bigquery_client │
                    │  BigQueryClient        │
                    │  (데이터 조회)           │
                    └────────────────────────┘
```

---

## 2. 상세 실행 흐름

### **Phase 1: 엔트리 포인트 (run_pipeline.py)**

```python
# 1. 경로 설정
project_root = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(project_root, 'app')
sys.path.insert(0, app_dir)

# 2. Orchestrator 함수 import
from routes.orchestrator import orchestrate_analysis

# 3. 분석 실행
result, agent_result = orchestrate_analysis(
    target_date="2025-11-10",
    context="일일 금융 시장 분석",
    return_agent_result=True
)
```

**역할:**
- 분석 기준일 설정 (하드코딩된 날짜 또는 동적 날짜)
- Orchestrator 호출
- 결과를 outputs 디렉토리에 저장 (`summary_*.txt`, `agent_result_*.txt`)

---

### **Phase 2: Orchestrator (routes/orchestrator.py)**

```python
def orchestrate_analysis(target_date, context, return_agent_result):
    # 1. LLMSummarizer 초기화 (지연 로딩)
    summarizer = get_llm_summarizer()
    
    # 2. LLM Agent를 통한 분석 실행
    result = summarizer.summarize(
        context=context,
        target_date=target_date
    )
    
    # 3. Agent 결과 파싱
    timeseries_prediction, sentiment_analysis = parse_agent_result(agent_result)
    
    # 4. OrchestratorOutput 생성 및 반환
    return OrchestratorOutput(...)
```

**역할:**
- LLM Summarizer 관리 (싱글톤 패턴)
- Agent 실행 결과를 구조화된 `OrchestratorOutput` 객체로 변환
- FastAPI 라우터로도 동작 가능 (`POST /api/orchestrator/summarize`)

---

### **Phase 3: LLM Summarizer (models/llm_summarizer.py)**

```python
class LLMSummarizer:
    def __init__(self):
        # 1. Vertex AI 인증 토큰 획득
        access_token = self._get_access_token()
        
        # 2. ChatOpenAI 인스턴스 생성 (Vertex AI OpenAI 호환 API)
        self.llm = ChatOpenAI(
            model="meta/llama-3.1-70b-instruct-maas",
            base_url="https://{location}-aiplatform.googleapis.com/v1/..."
        )
        
        # 3. LangChain Agent 생성 (Tools 바인딩)
        tools = [timeseries_predictor, news_sentiment_analyzer]
        self.agent = create_agent(model=llm_with_tools, tools=tools, system_prompt=SYSTEM_PROMPT)
    
    def summarize(self, context, target_date):
        # Agent 실행 → Tool 자동 호출 → 최종 보고서 생성
        result = self.agent.invoke({"messages": [HumanMessage(content=user_input)]})
```

**핵심 컴포넌트:**

| 컴포넌트 | 설명 |
|---------|------|
| `SYSTEM_PROMPT` | Agent에게 금융 분석가 역할 부여 + 보고서 형식 지정 |
| `REPORT_FORMAT` | 마크다운 형식의 상세 보고서 템플릿 |
| `@tool timeseries_predictor` | 시계열 예측 Tool |
| `@tool news_sentiment_analyzer` | 뉴스 감성분석 Tool |

---

### **Phase 4-A: 시계열 예측 Tool (models/timeseries_predictor.py)**

```python
@tool
def timeseries_predictor(target_date: str) -> str:
    return predict_market_trend(target_date)

def predict_market_trend(target_date):
    # 1. BigQuery에서 Prophet 피처 데이터 조회 (1500일치)
    bq_client = BigQueryClient()
    history_df = bq_client.get_prophet_features(target_date, lookback_days=1500)
    
    # 2. XGBoost 모델로 예측 수행
    engine = TimeSeriesInference()
    result = engine.predict(history_df, target_date)
    
    return json.dumps(result)  # JSON 형식으로 반환
```

**반환값 예시:**
```json
{
    "target_date": "2025-11-10",
    "forecast_value": 485.25,
    "forecast_direction": "Up",
    "confidence_score": 72.5,
    "recent_mean_7d": 480.12,
    "all_time_mean": 456.78,
    "trend_analysis": "Rising",
    "volatility_index": 5.23
}
```

---

### **Phase 4-B: 뉴스 감성분석 Tool (models/sentiment_analyzer.py)**

```python
@tool
def news_sentiment_analyzer(target_date: str) -> str:
    analyzer = SentimentAnalyzer()
    return json.dumps(analyzer.predict_market_impact(target_date))

class SentimentAnalyzer:
    def predict_market_impact(self, target_date):
        bq = BigQueryClient()
        
        # 1. BigQuery에서 데이터 조회
        news_df = bq.get_news_for_prediction(target_date, lookback_days=7)
        price_df = bq.get_price_history(target_date, lookback_days=30)
        
        # 2. 전처리 (임베딩 파싱 등)
        processed_news = preprocess_news_data(news_df)
        
        # 3. XGBoost 모델로 가격 영향도 예측 + 근거 뉴스 추출
        result = self.predictor.predict_with_evidence(
            news_data=processed_news,
            price_history=price_df,
            target_date=target_date,
            top_k=3  # 상위 3개 근거 뉴스
        )
        return result
```

**반환값 예시:**
```json
{
    "prediction": 1,
    "probability": 0.68,
    "evidence_news": [
        {
            "title": "옥수수 선물 급등...",
            "price_impact_score": 0.85,
            "all_text": "기사 본문...",
            "triples_text": "주체 관계 객체..."
        }
    ]
}
```

---

### **Phase 5: 데이터 레이어 (utils/bigquery_client.py)**

```python
class BigQueryClient:
    def get_prophet_features(target_date, lookback_days):
        """시계열 모델용 피처 데이터 (ds, y, yhat, trend 등)"""
        
    def get_news_for_prediction(target_date, lookback_days):
        """뉴스 데이터 (title, article_embedding, price_impact_score 등)"""
        
    def get_price_history(target_date, lookback_days):
        """가격 히스토리 데이터"""
```

---

## 3. 파일별 역할 요약

| 디렉토리/파일 | 역할 |
|-------------|------|
| **config/** | |
| settings.py | 환경변수 로드, 모델/API 설정 |
| **routes/** | |
| orchestrator.py | 분석 오케스트레이션, FastAPI 라우터 |
| **models/** | |
| llm_summarizer.py | Vertex AI + LangChain Agent, 보고서 생성 |
| timeseries_predictor.py | 시계열 예측 Tool (어댑터) |
| sentiment_analyzer.py | 뉴스 감성분석 Tool (어댑터) |
| **model/timeseries_model/** | |
| inference.py | XGBoost 모델 로드 및 예측 |
| **model/news_sentiment_model/** | |
| inference_with_evidence.py | 뉴스 기반 가격 예측 + 근거 추출 |
| **utils/** | |
| bigquery_client.py | BigQuery 데이터 조회 |
| **schema/** | |
| models.py | Pydantic 스키마 정의 |

---

## 4. 데이터 흐름 다이어그램

```
┌─────────────┐     target_date      ┌──────────────────┐
│ run_pipeline│ ─────────────────► │   Orchestrator   │
└─────────────┘                      └────────┬─────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │  LLMSummarizer   │
                                    │  (LangChain Agent)│
                                    └────────┬─────────┘
                                              │
                        ┌─────────────────────┼─────────────────────┐
                        │                     │                     │
                        ▼                     │                     ▼
              ┌─────────────────┐             │          ┌─────────────────────┐
              │ timeseries_tool │             │          │ news_sentiment_tool │
              └────────┬────────┘             │          └──────────┬──────────┘
                       │                      │                     │
                       ▼                      │                     ▼
              ┌─────────────────┐             │          ┌─────────────────────┐
              │   BigQuery      │             │          │     BigQuery        │
              │ (prophet_features)│            │          │ (news_article)      │
              └────────┬────────┘             │          └──────────┬──────────┘
                       │                      │                     │
                       ▼                      │                     ▼
              ┌─────────────────┐             │          ┌─────────────────────┐
              │ XGBoost 시계열  │             │          │ XGBoost 뉴스 모델   │
              │  (방향 예측)     │             │          │  (영향도 예측)       │
              └────────┬────────┘             │          └──────────┬──────────┘
                       │                      │                     │
                       │      JSON 결과       │      JSON 결과      │
                       └──────────────────────┼─────────────────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │   Vertex AI LLM  │
                                    │ (Llama 3.1 70B)  │
                                    │   보고서 생성     │
                                    └────────┬─────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │  최종 보고서      │
                                    │  (Markdown)      │
                                    └──────────────────┘
```

---

## 5. 핵심 설계 패턴

1. **LangChain Agent 패턴**: LLM이 자율적으로 Tool을 선택하고 호출
2. **Adapter 패턴**: `models/` 폴더의 클래스들이 실제 모델(`model/`)을 래핑
3. **Lazy Initialization**: 모델/클라이언트를 필요할 때만 초기화
4. **Repository 패턴**: BigQueryClient가 데이터 접근 추상화