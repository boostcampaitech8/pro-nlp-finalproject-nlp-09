from typing import Optional
from langchain_core.tools import tool
import subprocess
import json
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


# 상수 정의
REPORT_FORMAT = """**일일 금융 시장 분석 보고서 **
> **📅 분석 일자 ** : (YYYY-MM-DD)
> **💬 종합 의견 ** : [종합 의견 한줄 요약]
---

### 1. 📊 시계열 데이터 분석가 의견
> [시계열 데이터 분석 한줄 평가]

| 항목 | 내용 |
|------|------|
| **입력 데이터 길이** | 5년 (Prophet Features) |
| **마지막 관측값** | [Last Observed Value] |
| **시계열 예측값** | [Forecast Value] |
| **신뢰도** | [Confidence Score] % |

- **추세 분석**
  - **최근 기간 평균** (7일) : [Recent Mean]
  - **전 기간 평균** : [All-time Mean]
  - **최근 변동 추이** : [Trend Analysis: Rising/Falling 등 설명]
  - **시계열 예측값 해석** : [Forecast Direction] 방향으로 예측되며, 신뢰도는 [Confidence Score]% 입니다.

- **예측값 해석**
  - **현재 수준 대비** : [Last Value] 대비 [Forecast Value] 로 변동 예상.
  - **단기 변동성 평가** : 변동성 지표 [Volatility Index] 수준.

---

### 2. 📰 뉴스 감성분석 결과 분석
> [뉴스 기사 감성분석 한줄 평가]

- **주요 뉴스 (evidence_news)**
  - news_sentiment_analyzer 도구 결과를 반드시 아래 표 형식으로 표시하세요.
  - **중요**: title과 all_text가 영어로 되어 있으면 반드시 한국어로 번역하여 표시하세요.
  
  | 번호 | 뉴스 제목 | 영향력 점수 | 요약 |
  |-----------|-----------|-------------|------|
  | 1 | [뉴스 제목(한국어 번역)] | [점수] | [내용 요약] |
  | 2 | [뉴스 제목(한국어 번역)] | [점수] | [내용 요약] |
  | ... | ... | ... | ... |

- **시장 영향력 분석**
  - **상승 확률**: [Probability] %
  - **종합 의견**: [뉴스 기반 상승/하락 예측 의견]

- **텍스트적 근거**
  - [각 뉴스가 시장에 미치는 영향 분석]
  - **주요 키워드**: [keyword_analyzer 결과의 top_entities 상위 10개 entity]

- **과거 관련 뉴스**
  - pastnews_rag 도구 결과를 반드시 아래 표 형식으로 표시하세요.
  - **중요**: description이 영어로 되어 있으면 반드시 한국어로 번역하여 "뉴스 제목" 컬럼에 표시하세요.
  
  | 뉴스 날짜 | 뉴스 내용 | 당일 | 1일후 | 3일후 |
  |-----------|-----------|------|------|------|
  | [뉴스 날짜] | [뉴스 내용(한국어 번역)] | [price_0일후] | [price_1일후] | [price_3일후] |
  | [뉴스 날짜] | [뉴스 내용(한국어 번역)] | [price_0일후] | [price_1일후] | [price_3일후] |
  | ... | ... | ... | ... | ... |

---

### 3. 종합 의견

- **[현재 시장 상황 요약]**
- **[주요 지표 및 뉴스 요약]**
- **[단기/중기/장기 전망 요약]**
- **[투자자 입장에서의 조언]**

**결론**: [날짜] 기준, 시장은 **[전망]**을 유지할 것으로 전망되며, **[주요 성장 동력]**이 주요 성장 동력입니다. 그러나 **[주요 리스크]**에 따른 리스크를 주의 깊게 모니터링해야 합니다.

**중요**: 
- 반드시 위 형식을 정확히 따라야 합니다.
- 표 형식은 마크다운 테이블 문법을 사용하세요.
- 섹션 번호와 제목은 정확히 일치해야 합니다.
- 각 섹션은 "---"로 구분하세요.
- 주요 키워드는 #키워드1 #키워드2 형식으로 표기
- 뉴스 관련 내용이 영어로 되어 있으면 반드시 한국어로 번역하여 표시하세요. 원문을 그대로 표시하지 마세요.
- 언어는 반드시 순수 한국어(한글)만 사용하세요."""

SYSTEM_PROMPT = (
    """당신은 전문 금융 분석가입니다.

**사용 가능한 도구**:
1. timeseries_predictor: 시계열 데이터 기반 시장 예측
   - target_date: 분석할 대상 날짜 (형식: "YYYY-MM-DD")
   - 설명: 지정된 날짜의 가격 추세, 예측값, 신뢰도 등을 반환합니다.

2. news_sentiment_analyzer: 뉴스 기반 시장 영향력 분석 및 근거 추출
   - target_date: 분석할 대상 날짜 (형식: "YYYY-MM-DD")
   - 설명: 해당 날짜 전후의 뉴스를 분석하여 시장 상승/하락 확률을 예측하고, 예측의 핵심 근거가 된 주요 뉴스들을 반환합니다.

3. keyword_analyzer: 뉴스 기사의 주요 키워드 분석 (Entity Confidence / PageRank 기반)
   - target_date: 분석할 대상 날짜 (형식: "YYYY-MM-DD")
   - days: 분석할 일수 (기본 3일)
   - 설명: PageRank 알고리즘을 활용하여 뉴스의 Entity Confidence(중요도) 상위 키워드를 추출합니다.
   - 반환 값: top_entities (상위 10개, 각 항목: {"entity": "...", "score": ...})

4. pastnews_rag: 전달받은 triples로 유사 뉴스 description과 publish_date 조회
   - triples_json: keyword_analyzer 결과의 top_triples에서 각 항목의 "triple" 배열만 모은 JSON 문자열. 예: [["United States","experiencing","government shutdown"], ...]
   - top_k: 유사 hash_id 개수 (기본 5)
   - 설명: keyword_analyzer 호출 후, 그 결과의 top_triples를 triples_json 인자로 넘겨서 호출하세요. 유사한 triple을 가진 뉴스 기사의 description과 publish_date를 반환합니다.

**도구 사용 규칙**:
- 분석 대상 날짜(target_date)가 주어지면 반드시 `timeseries_predictor`, `news_sentiment_analyzer`, `keyword_analyzer`를 모두 호출한 뒤, keyword_analyzer 결과의 top_triples를 triples_json 인자로 넘겨 `pastnews_rag(triples_json=..., top_k=5)`를 한 번 호출하세요.
- 이전 도구가 오류를 반환하더라도, 세 도구를 반드시 모두 호출한 뒤에만 보고서를 작성하세요.
- `news_sentiment_analyzer` 결과에 포함된 'evidence_news'는 보고서의 '### 2. 📰 뉴스 감성분석 결과 분석' 섹션의 '주요 뉴스 (evidence_news)' 항목에 아래 표 형식으로 표시하세요. **title과 all_text가 영어로 되어 있으면 반드시 한국어로 번역하여 표시하세요.**
  | 번호 | 뉴스 제목 | 영향력 점수 | 요약 |
  |-----------|-----------|-------------|------|
  | [번호] | [뉴스 제목(한국어 번역)] | [점수] | [내용 요약(한국어 번역)] |
- `pastnews_rag` 도구 결과(article_info)는 반드시 '### 2. 📰 뉴스 감성분석 결과 분석' 섹션 내 '과거 관련 뉴스 (pastnews_rag)' 항목에 아래 표 형식으로 표시하세요. **description이 영어로 되어 있으면 반드시 한국어로 번역하여 "뉴스 제목" 컬럼에 표시하세요.**
  | 뉴스 날짜 | 뉴스 제목 | 당일 | 1일후 | 3일후 |
  |-----------|-----------|------|------|------|
  | [뉴스 날짜] | [뉴스 제목(한국어 번역)] | [price_0일후] | [price_1일후] | [price_3일후] |
- `keyword_analyzer` 결과의 top_entities를 활용할 때: (1) score는 사용하지 마세요. (2) entity 이름만 사용하여 최대한 한국어로 번역하세요. (3) #키워드1 #키워드2 형식으로 표기하세요. 예: #옥수수 #가격 #수출 #미국농무부 #시장
- 세 도구의 결과를 종합하여 논리적인 금융 보고서를 작성하세요. 시계열 지표, 뉴스 감성 분석, 키워드 분석 결과가 서로 보완되도록 서술하세요.
- target_date는 반드시 다음 문자열 리터럴을 그대로 복사해서 사용하세요. (YYYY-MM-DD)
**보고서 작성 형식 (반드시 이 형식을 따라야 합니다)**:

"""
    + REPORT_FORMAT
)


# LangChain Tools 정의
@tool
def timeseries_predictor(target_date: str) -> str:
    """
    특정 날짜의 금융 시장 추세(상승/하락)와 가격을 예측합니다.

    Args:
        target_date: 분석할 날짜 문자열 (형식: "YYYY-MM-DD")

    Returns:
        JSON 형식의 예측 결과 문자열 (예측값, 방향, 신뢰도, 추세 분석 등 포함)
    """
    return predict_market_trend(target_date)


@tool
def news_sentiment_analyzer(target_date: str) -> str:
    """
    특정 날짜의 뉴스를 분석하여 시장 영향력을 예측하고 주요 근거 뉴스(제목, 영향력 점수, 관계 정보 등)를 제공합니다.

    Args:
        target_date: 분석할 날짜 문자열 (형식: "YYYY-MM-DD")

    Returns:
        JSON 형식의 예측 결과 문자열 (상승 확률, 근거 뉴스 리스트, 피처 요약 포함)
    """
    analyzer = SentimentAnalyzer()
    result = analyzer.predict_market_impact(target_date)
    return json.dumps(result, ensure_ascii=False)


@tool
def keyword_analyzer(target_date: str, days: int = 3) -> str:
    """
    특정 날짜 기준으로 뉴스 기사의 주요 키워드를 분석합니다.
    PageRank 알고리즘(Entity Confidence)과 임베딩 기반 클러스터링을 활용하여 핵심 엔티티를 추출합니다.

    Args:
        target_date: 분석할 날짜 문자열 (형식: "YYYY-MM-DD")
        days: 분석할 일수 (기본 3일, 최대 7일 권장)

    Returns:
        JSON: top_entities (상위 10개), top_triples (핵심 엔티티가 포함된 triple 중 엣지 실제 weight×entity PageRank 중요도 상위 10개, 각 항목: {"triple": [s,v,o], "importance": 점수})
    """
    result = json.loads(_analyze_keywords(target_date=target_date, days=days, top_k=10))
    top_entities = result.get("top_entities", [])[:10]
    top_triples = result.get("top_triples", [])
    return json.dumps({"top_entities": top_entities, "top_triples": top_triples}, ensure_ascii=False, indent=2)


@tool
def pastnews_rag(triples_json: str, top_k: int = 5) -> str:
    """
    전달받은 triples(triples_json)로 유사 뉴스 hash_id 검색 및 해당 뉴스의 description과 publish_date를 조회합니다.
    keyword_analyzer 호출 후, 그 결과의 top_triples에서 각 항목의 "triple" 배열만 모아 JSON 문자열로 넘기세요.

    Args:
        triples_json: triples 배열의 JSON 문자열. 각 triple은 [주어, 동사, 목적어]. 예: [["United States","experiencing","government shutdown"], ...]
        top_k: 유사 hash_id 개수 (기본 5)

    Returns:
        JSON: article_info (각 항목: {"description": str, "publish_date": str, "price_0일후": float, "price_1일후": float, "price_3일후": float}), error(있을 경우)
    """
    triples = []
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
    result = _run_pastnews_rag(triples=triples if triples else None, top_k=top_k)
    return json.dumps(result, ensure_ascii=False, indent=2)


class LLMSummarizer:
    """Vertex AI를 사용하는 LangChain Agent를 이용한 통합 분석"""

    def __init__(self, model_name: str = None, project_id: str = None, location: str = None):
        """
        Args:
            model_name: 생성 모델 이름 (기본값: 설정 파일의 GENERATE_MODEL_NAME)
            project_id: Google Cloud 프로젝트 ID (지정하지 않으면 설정 파일 또는 gcloud config에서 자동으로 가져옴)
            location: Vertex AI 리전 (기본값: 설정 파일의 VERTEX_AI_LOCATION)
        """
        self.model_name = model_name or GENERATE_MODEL_NAME
        self.project_id = project_id or VERTEX_AI_PROJECT_ID or self._get_project_id()
        self.location = location or VERTEX_AI_LOCATION
        self.llm = None
        self.agent = None
        self._initialize()

    # TODO project id .env로 관리
    def _get_project_id(self) -> str:
        """gcloud config에서 프로젝트 ID를 가져옴"""
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            else:
                raise ValueError("gcloud config에서 프로젝트를 찾을 수 없습니다.")
        except Exception as e:
            raise ValueError(
                f"project_id가 필요합니다.\n"
                f"해결 방법: gcloud config set project YOUR_PROJECT_ID\n"
                f"또는 환경변수 GOOGLE_CLOUD_PROJECT를 설정하세요.\n"
                f"오류: {e}"
            )

    def _create_llm(self) -> ChatVertexAI:
        """ChatVertexAI 인스턴스 생성 (모델명·프로젝트·리전은 env에서 로드)"""
        return ChatVertexAI(
            model=self.model_name,
            project=self.project_id,
            location=self.location,
            temperature=GENERATE_MODEL_TEMPERATURE,
            max_output_tokens=GENERATE_MODEL_MAX_TOKENS,
        )

    def _initialize(self):
        """LLM 및 Agent 초기화"""
        self.llm = self._create_llm()
        print(f"✅ ChatVertexAI 사용 (모델: {self.model_name}, env 기반)")

        tools = [timeseries_predictor, news_sentiment_analyzer, keyword_analyzer, pastnews_rag]
        llm_with_tools = self.llm.bind_tools(tools)

        self.agent = create_agent(
            model=llm_with_tools,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )

    def _build_user_input(
        self,
        context: str,
        target_date: str,
    ) -> str:
        """Agent에게 전달할 사용자 입력 메시지 생성"""

        user_input = f"""다음 정보를 바탕으로 전문적인 금융 시장 분석 보고서를 작성해주세요.

**분석 맥락**: {context or "최근 시장 상황 분석"}
**분석 기준 일자**: {target_date}

위의 시스템 프롬프트에 명시된 규칙과 형식을 따라 보고서를 작성해주세요.
"""
        return user_input

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
1. 섹션 제목이 정확히 일치해야 합니다: "### 1. 📊 시계열 데이터 분석가 의견", "### 2. 📰 뉴스 감성분석 결과 분석", "### 3. 종합 의견"
2. 각 섹션은 "---"로 구분되어야 합니다 (최소 3개)
3. 마크다운 테이블 형식(|)을 사용해야 합니다
4. 헤더에 "📅 분석 일자"와 "💬 종합 의견"이 포함되어야 합니다
5. Tool 호출 후 반드시 최종 보고서를 작성해야 합니다"""
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
