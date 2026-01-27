from typing import Optional, List
from langchain_core.tools import tool
import subprocess
from google.auth import default
from google.auth.transport.requests import Request
from langchain_core.messages import HumanMessage, AIMessage

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config.settings import (
    GENERATE_MODEL_NAME, GENERATE_MODEL_TEMPERATURE, GENERATE_MODEL_MAX_TOKENS,
    VERTEX_AI_PROJECT_ID, VERTEX_AI_LOCATION,
)
from models.timeseries_predictor import TimeSeriesPredictor
from models.sentiment_analyzer import SentimentAnalyzer


# 상수 정의
REPORT_FORMAT = """**일일 금융 시장 분석 보고서 **
> **📅 분석 일자 ** : (YYYY-MM-DD)
> **💬 종합 의견 ** : [종합 의견 한줄 요약]
---

### 1. 📊 시계열 데이터 분석가 의견
> [시계열 데이터 분석 한줄 평가]

| 항목 | 내용 |
|------|------|
| **입력 데이터 길이** | X일 (YYYY-MM-DD 부터 YYYY-MM-DD까지) |
| **마지막 관측값** | XXX.XX |
| **시계열 예측값** | XXX.XX |
| **신뢰도** | XX.XX % |

- **추세 분석**
  - [최근 기간 평균과 전 기간 평균 비교]
  - [최근 변동 추이 설명]
  - [시계열 예측값 해석 및 신뢰도 평가]

- **예측값 해석**
  - [현재 수준 대비 예측값 의미]
  - [단기 변동성 평가]

---

### 2. 📰 뉴스 감성분석 결과 분석
> [뉴스 기사 감성분석 한줄 평가]

| 기사 번호 | 내용 요약 | 감성 |
|-----------|-----------|------|
| 1 | [기사 요약] | 긍정/부정/중립 |
| 2 | [기사 요약] | 긍정/부정/중립 |
| ... | ... | ... |

- **감성 비율**
  - 긍정: X개 (XX %)
  - 부정: X개 (XX %)
  - 중립: X개 (XX %)
  - **종합 감성**: 긍정/부정/중립

- **텍스트적 근거**
  - [각 기사가 시장에 미치는 영향 분석]

---

### 3. 미래 시장 전망

| 구분 | 근거 | 전망 |
|------|------|------|
| **단기(1–3일)** | [근거 요약] | **[전망]** [상세 설명] |
| **중기(1주)** | [근거 요약] | **[전망]** [상세 설명] |
| **장기(1개월)** | [근거 요약] | **[전망]** [상세 설명] |

- **위험 요인**
  - [주요 위험 요인 나열]

- **기회 요인**
  - [주요 기회 요인 나열]

---

### 4. 종합 의견

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
- 언어는 반드시 순수 한국어(한글)만 사용하세요."""

SYSTEM_PROMPT = """당신은 전문 금융 분석가입니다.

**사용 가능한 도구**:
1. timeseries_predictor: 시계열 데이터 예측
   - data: 쉼표로 구분된 숫자 문자열 (예: "100.5,101.2,102.3")

2. news_sentiment_analyzer: 뉴스 기사 감성분석
   - texts: "[기사 1]\\n내용\\n\\n[기사 2]\\n내용" 형식의 뉴스 기사 문자열

**도구 사용 규칙**:
- 사용자 입력에 제공된 시계열 데이터를 timeseries_predictor 도구에 전달하여 예측을 수행하세요.
- 사용자 입력에 제공된 뉴스 기사를 news_sentiment_analyzer 도구에 전달하여 감성분석을 수행하세요.
- 각 도구는 필요한 만큼 사용하세요.
- 도구 실행 결과를 받은 후 종합 분석을 수행하세요.

**보고서 작성 형식 (반드시 이 형식을 따라야 합니다)**:

""" + REPORT_FORMAT


# LangChain Tools 정의
@tool
def timeseries_predictor(data: str) -> str:
    """
    시계열 데이터를 예측합니다.
    
    Args:
        data: 쉼표로 구분된 숫자 문자열 (예: "100.5,101.2,102.3")
    
    Returns:
        예측값과 신뢰도를 포함한 분석 결과
    """
    data_list = [float(x.strip()) for x in data.split(",") if x.strip()]
    predictor = TimeSeriesPredictor()
    prediction, confidence = predictor.predict(data_list)
    
    result = f"""
시계열 예측 결과:
- 예측값: {prediction:.2f}
- 신뢰도: {confidence:.2%}
- 입력 데이터 길이: {len(data_list)}
"""
    return result.strip()


def _format_sentiment_results(text_list: List[str], results: List[dict]) -> str:
    """감성 분석 결과를 포맷팅하여 반환"""
    sentiment_map = {"positive": "긍정", "negative": "부정", "neutral": "중립"}
    
    # 감성별 개수 계산
    counts = {
        "positive": sum(1 for r in results if r.get("sentiment") == "positive"),
        "negative": sum(1 for r in results if r.get("sentiment") == "negative"),
        "neutral": sum(1 for r in results if r.get("sentiment") == "neutral"),
    }
    total = len(results)
    
    # 기사별 상세 결과 생성
    detailed_results = []
    for i, (text, result) in enumerate(zip(text_list, results), 1):
        sentiment_en = result.get("sentiment", "neutral")
        sentiment_ko = sentiment_map.get(sentiment_en, "중립")
        detailed_results.append(f"기사 {i}: [{sentiment_ko}] {text}")
    
    # 종합 감성 결정
    if counts["positive"] > counts["negative"]:
        overall = "긍정"
    elif counts["negative"] > counts["positive"]:
        overall = "부정"
    else:
        overall = "중립"
    
    return f"""뉴스 감성분석 결과:
- 분석된 기사 수: {total}개
- 긍정: {counts['positive']}개 ({counts['positive']/total*100:.1f}%)
- 부정: {counts['negative']}개 ({counts['negative']/total*100:.1f}%)
- 중립: {counts['neutral']}개 ({counts['neutral']/total*100:.1f}%)
- 종합 감성: {overall}

기사별 감성 분석:
{chr(10).join(detailed_results)}
""".strip()


@tool
def news_sentiment_analyzer(texts: str) -> str:
    """
    FinBERT 모델을 사용하여 뉴스 기사들의 감성을 분석합니다.
    
    Args:
        texts: "[기사 1]\\n내용\\n\\n[기사 2]\\n내용" 형식의 뉴스 기사 문자열
    
    Returns:
        각 기사의 감성 분석 결과와 종합 감성
    """
    import re
    
    # [기사 N] 패턴으로 기사 추출 (라벨과 내용을 함께 하나의 기사로 인식)
    article_pattern = r'\[기사\s*\d+\]\s*\n(.+?)(?=\n\n\[기사\s*\d+\]|$)'
    matches = re.finditer(article_pattern, texts, re.DOTALL)
    
    text_list = []
    for match in matches:
        article_text = match.group(1).strip()
        if article_text:
            text_list.append(article_text)
    
    if not text_list:
        return "분석할 기사가 없습니다."
    
    analyzer = SentimentAnalyzer()
    results = analyzer.analyze_batch(text_list)
    return _format_sentiment_results(text_list, results)


class LLMSummarizer:
    """Vertex AI를 사용하는 LangChain Agent를 이용한 통합 분석"""
    
    def __init__(
        self, 
        model_name: str = None,
        project_id: str = None,
        location: str = None
    ):
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
    
    def _get_project_id(self) -> str:
        """gcloud config에서 프로젝트 ID를 가져옴"""
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True,
                timeout=2
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
    
    def _get_access_token(self) -> str:
        """Google Cloud 인증 토큰 가져오기"""
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not credentials.valid:
            credentials.refresh(Request())
        return credentials.token
    
    def _build_base_url(self) -> str:
        """Vertex AI OpenAI 호환 API base URL 생성"""
        return (
            f"https://{self.location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{self.location}/endpoints/openapi"
        )
    
    def _create_llm(self, access_token: str) -> ChatOpenAI:
        """ChatOpenAI 인스턴스 생성"""
        return ChatOpenAI(
            model=self.model_name,
            base_url=self._build_base_url(),
            api_key=access_token,
            temperature=GENERATE_MODEL_TEMPERATURE,
            max_tokens=GENERATE_MODEL_MAX_TOKENS,
            model_kwargs={
                "parallel_tool_calls": False,
            },
        )
    
    
    def _initialize(self):
        """LLM 및 Agent 초기화"""
        access_token = self._get_access_token()
        self.llm = self._create_llm(access_token)
        print(f"✅ ChatOpenAI (Vertex AI OpenAI 호환 API) 사용: {self.model_name}")
        
        tools = [
            timeseries_predictor,
            news_sentiment_analyzer
        ]
        llm_with_tools = self.llm.bind_tools(tools)
        
        self.agent = create_agent(
            model=llm_with_tools,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )
    
    def _build_user_input(
        self,
        context: str,
        timeseries_table_id: Optional[str] = None,
        timeseries_value_column: Optional[str] = None,
        timeseries_days: Optional[int] = None,
        news_table_id: Optional[str] = None,
        news_value_column: Optional[str] = None,
        news_days: Optional[int] = None,
        timeseries_data: Optional[List[float]] = None,
        news_texts: Optional[List[str]] = None
    ) -> str:
        """Agent에게 전달할 사용자 입력 메시지 생성
        
        Args:
            context: 분석 맥락
            timeseries_data: 직접 전달할 시계열 데이터
            news_texts: 직접 전달할 뉴스 텍스트
            나머지 파라미터는 하위 호환성을 위해 유지하지만 사용하지 않음
        """
        user_input = f"""다음 정보를 바탕으로 전문적인 금융 시장 분석 보고서를 작성해주세요.

**분석 맥락**: {context or "최근 시장 상황 분석"}

"""
        
        # 시계열 데이터 직접 포함
        if timeseries_data:
            data_str = ", ".join(map(str, timeseries_data))
            user_input += f"**시계열 데이터**: {data_str}\n\n"
            user_input += "- 이 데이터를 timeseries_predictor 도구에 전달하여 예측을 수행하세요.\n\n"
        
        # 뉴스 기사 직접 포함
        if news_texts:
            texts_str = "\n\n".join([f"[기사 {i+1}]\n{text}" for i, text in enumerate(news_texts)])
            user_input += f"**분석할 뉴스 기사**:\n{texts_str}\n\n"
            user_input += "- 이 데이터를 news_sentiment_analyzer 도구에 전달하여 감성분석을 수행하세요.\n"
        
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
    
    def _extract_summary_from_result(self, result: dict) -> str:
        """Agent 실행 결과에서 요약 텍스트 추출"""
        import json
        messages = result.get("messages", [])
        
        # messages에서 마지막 AIMessage의 content 추출
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = str(msg.content) if msg.content else ""
                content = content.strip().rstrip('\\')
                
                # JSON 형식의 tool call arguments는 건너뛰기
                if content.startswith("{") and content.strip().endswith("}"):
                    try:
                        # JSON 파싱 시도
                        parsed = json.loads(content)
                        # tool call arguments 형식인지 확인 (texts, data 등의 키가 있으면 건너뛰기)
                        if isinstance(parsed, dict) and any(key in parsed for key in ["texts", "data"]):
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
            return str(output).strip().rstrip('\\')
        
        # 모든 방법 실패 시 전체 결과를 문자열로 변환
        return str(result).strip().rstrip('\\')
    
    def summarize(
        self,
        context: str = "",
        timeseries_table_id: Optional[str] = None,
        timeseries_value_column: Optional[str] = None,
        timeseries_days: Optional[int] = None,
        news_table_id: Optional[str] = None,
        news_value_column: Optional[str] = None,
        news_days: Optional[int] = None,
        timeseries_data: Optional[List[float]] = None,
        news_texts: Optional[List[str]] = None,
        max_retries: int = 2,
    ) -> dict:
        """LangChain Agent를 이용한 LLM 요약 생성
        
        Args:
            context: 분석 맥락 (시간 범위, 시장 상황 등)
            timeseries_table_id: 시계열 데이터 테이블명 (기본값: "corn_price")
            timeseries_value_column: 시계열 값 컬럼명 (기본값: "close")
            timeseries_days: 시계열 데이터 가져올 일수 (기본값: 30)
            news_table_id: 뉴스 테이블명 (기본값: "news_article")
            news_value_column: 뉴스 텍스트 컬럼명 (기본값: "description")
            news_days: 뉴스 가져올 일수 (기본값: 3)
            timeseries_data: 시계열 예측에 사용할 데이터 (하위 호환성, 권장하지 않음)
            news_texts: 감성분석에 사용할 뉴스 텍스트 리스트 (하위 호환성, 권장하지 않음)
            max_retries: 출력 형식이 맞지 않을 때 최대 재시도 횟수 (기본값: 2)
        
        Returns:
            dict: {
                'summary': str,  # LLM 요약
                'agent_result': dict,  # Agent 실행 결과 전체 (Tool 메시지 포함)
            }
        """
        user_input = self._build_user_input(
            context=context,
            timeseries_table_id=timeseries_table_id,
            timeseries_value_column=timeseries_value_column,
            timeseries_days=timeseries_days,
            news_table_id=news_table_id,
            news_value_column=news_value_column,
            news_days=news_days,
            timeseries_data=timeseries_data,
            news_texts=news_texts
        )
        
        for attempt in range(max_retries + 1):
            # Agent 실행 (LangChain이 자동으로 tool call을 처리함)
            if attempt == 0:
                result = self.agent.invoke({
                    "messages": [HumanMessage(content=user_input)]
                })
            else:
                # 재시도 시 기존 메시지 사용
                result = self.agent.invoke({
                    "messages": result.get('messages', [])
                })
            
            # 결과 추출
            if isinstance(result, dict):
                messages = result.get('messages', [])
                
                summary = self._extract_summary_from_result(result)
                agent_result = result
                
                # 디버깅: 메시지 상태 확인
                print(f"\n[디버깅] 총 메시지 수: {len(messages)}")
                tool_call_count = sum(1 for msg in messages if isinstance(msg, AIMessage) and msg.tool_calls)
                tool_result_count = sum(1 for msg in messages if hasattr(msg, 'name') and msg.name)
                print(f"  Tool 호출: {tool_call_count}회, Tool 결과: {tool_result_count}개")
            else:
                summary = str(result).strip().rstrip('\\')
                agent_result = {'messages': []}
            
            # 요약이 비어있거나 너무 짧은 경우 확인
            if not summary or len(summary.strip()) < 50:
                print(f"\n⚠️ 요약이 비어있거나 너무 짧습니다 (길이: {len(summary)}자)")
                # 마지막 AIMessage에서 실제 텍스트 찾기
                if isinstance(result, dict):
                    messages = result.get('messages', [])
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage) and msg.content:
                            content = str(msg.content)
                            # GPT-OSS-20B 특수 형식이 아닌 실제 텍스트 찾기
                            if "<|channel|>" not in content and len(content.strip()) > 50:
                                summary = content.strip()
                                print(f"  → 대체 텍스트 발견 (길이: {len(summary)}자)")
                                break
            
            # 출력 형식 검증
            if summary and len(summary.strip()) > 50 and self._validate_output_format(summary):
                return {
                    'summary': summary or '',
                    'agent_result': agent_result,
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
1. 섹션 제목이 정확히 일치해야 합니다: "### 1. 📊 시계열 데이터 분석가 의견", "### 2. 📰 뉴스 감성분석 결과 분석", "### 3. 미래 시장 전망", "### 4. 종합 의견"
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
            'summary': summary or '',
            'agent_result': agent_result,
        }