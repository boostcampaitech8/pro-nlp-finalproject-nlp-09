from fastapi import APIRouter
from schema.models import OrchestratorInput, OrchestratorOutput, TimeSeriesPrediction, SentimentAnalysis
from models.timeseries_predictor import TimeSeriesPredictor
from models.sentiment_analyzer import SentimentAnalyzer
from models.llm_summarizer import LLMSummarizer
from datetime import datetime
from typing import Optional, Tuple, List, Union
import re

router = APIRouter(prefix="/api/orchestrator", tags=["orchestrator"])

# 모델 초기화 (Lazy initialization)
timeseries_predictor = None
sentiment_analyzer = None
llm_summarizer = None

def get_llm_summarizer():
    """LLM Summarizer 지연 초기화"""
    global llm_summarizer
    if llm_summarizer is None:
        llm_summarizer = LLMSummarizer()
    return llm_summarizer


def parse_agent_result(agent_result: dict) -> Tuple[TimeSeriesPrediction, list]:
    """
    Agent 실행 결과에서 Tool 메시지를 파싱하여 구조화된 데이터 반환
    
    역할: Agent 실행 결과 파싱 및 구조화 (orchestration 로직)
    
    Args:
        agent_result: Agent 실행 결과 dict (messages 포함)
    
    Returns:
        tuple: (timeseries_prediction, sentiment_analysis)
    """
    from langchain_core.messages import ToolMessage
    
    messages = agent_result.get('messages', []) if isinstance(agent_result, dict) else []
    
    timeseries_prediction = None
    sentiment_analysis = []
    
    for msg in messages:
        if isinstance(msg, ToolMessage):
            if msg.name == "timeseries_predictor":
                # 시계열 예측 결과 파싱
                timeseries_text = msg.content
                prediction_match = re.search(r'예측값:\s*([\d.]+)', timeseries_text)
                confidence_match = re.search(r'신뢰도:\s*([\d.]+)%', timeseries_text)
                
                if prediction_match and confidence_match:
                    timeseries_prediction = TimeSeriesPrediction(
                        prediction=float(prediction_match.group(1)),
                        confidence=float(confidence_match.group(1)) / 100,
                        timestamp=datetime.now().isoformat()
                    )
            
            elif msg.name == "news_sentiment_analyzer":
                # 감성분석 결과 파싱
                sentiment_text = msg.content
                
                # 기사별 감성 분석 추출
                # 패턴: "기사 1: [긍정] 텍스트 내용"
                article_pattern = r'기사\s*(\d+):\s*\[(긍정|부정|중립)\]\s*(.+?)(?=\n기사\s*\d+:|$)'
                matches = re.finditer(article_pattern, sentiment_text, re.DOTALL)
                
                sentiment_map = {"긍정": "positive", "부정": "negative", "중립": "neutral"}
                
                for match in matches:
                    sentiment_ko = match.group(2)
                    text = match.group(3).strip()
                    
                    sentiment_en = sentiment_map.get(sentiment_ko, "neutral")
                    
                    # 점수 추출 (기본값)
                    scores = {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
                    if sentiment_en == "positive":
                        scores = {"positive": 0.7, "negative": 0.2, "neutral": 0.1}
                    elif sentiment_en == "negative":
                        scores = {"positive": 0.2, "negative": 0.7, "neutral": 0.1}
                    
                    sentiment_analysis.append(
                        SentimentAnalysis(
                            text=text,
                            sentiment=sentiment_en,
                            scores=scores
                        )
                    )
    
    # 기본값 설정 (Tool 결과가 없는 경우)
    if not timeseries_prediction:
        timeseries_prediction = TimeSeriesPrediction(
            prediction=0.0,
            confidence=0.0,
            timestamp=datetime.now().isoformat()
        )
    
    return timeseries_prediction, sentiment_analysis


def orchestrate_analysis(
    timeseries_data: Optional[List[float]] = None,
    news_articles: Optional[List[str]] = None,
    context: str = "금융 시장 분석",
    timeseries_table_id: Optional[str] = None,
    timeseries_value_column: Optional[str] = None,
    timeseries_days: Optional[int] = None,
    news_table_id: Optional[str] = None,
    news_value_column: Optional[str] = None,
    news_days: Optional[int] = None,
    return_agent_result: bool = False
) -> Union[OrchestratorOutput, Tuple[OrchestratorOutput, dict]]:
    """
    Orchestrator 분석 로직 (동기 함수)
    
    HTTP 엔드포인트 없이 직접 호출 가능한 버전
    run_pipeline.py나 다른 스크립트에서 직접 사용 가능
    
    Args:
        timeseries_data: 시계열 데이터 리스트 (하위 호환성, 권장하지 않음)
        news_articles: 뉴스 기사 텍스트 리스트 (하위 호환성, 권장하지 않음)
        context: 분석 맥락 (기본값: "금융 시장 분석")
        timeseries_table_id: 시계열 데이터 테이블명 (기본값: "corn_price")
        timeseries_value_column: 시계열 값 컬럼명 (기본값: "close")
        timeseries_days: 시계열 데이터 가져올 일수 (기본값: 30)
        news_table_id: 뉴스 테이블명 (기본값: "news_article")
        news_value_column: 뉴스 텍스트 컬럼명 (기본값: "description")
        news_days: 뉴스 가져올 일수 (기본값: 3)
        return_agent_result: True면 (OrchestratorOutput, agent_result) 튜플 반환
    
    Returns:
        OrchestratorOutput: 구조화된 분석 결과
        또는 (OrchestratorOutput, dict): 구조화된 분석 결과와 agent_result
    """
    # 1. 입력 데이터 검증 및 준비
    timeseries_data_input = timeseries_data if timeseries_data else None
    news_texts = news_articles if news_articles else None
    
    # 2. LLMSummarizer 호출 (Agent 실행)
    summarizer = get_llm_summarizer()
    result = summarizer.summarize(
        context=context,
        timeseries_table_id=timeseries_table_id,
        timeseries_value_column=timeseries_value_column,
        timeseries_days=timeseries_days,
        news_table_id=news_table_id,
        news_value_column=news_value_column,
        news_days=news_days,
        timeseries_data=timeseries_data_input,
        news_texts=news_texts
    )
    
    # 3. Agent 실행 결과 파싱 (Tool 메시지 추출)
    agent_result = result.get('agent_result', {})
    timeseries_prediction, sentiment_analysis = parse_agent_result(agent_result)
    
    # 4. 최종 응답 구성
    output = OrchestratorOutput(
        timeseries_prediction=timeseries_prediction,
        sentiment_analysis=sentiment_analysis,
        llm_summary=result.get('summary', '')
    )
    
    if return_agent_result:
        return output, agent_result
    return output


@router.post("/summarize", response_model=OrchestratorOutput)
async def orchestrate_summary(input_data: OrchestratorInput):
    """
    FastAPI 엔드포인트: 파이프라인 조율 및 결과 구성
    
    내부적으로 orchestrate_analysis() 함수를 호출
    
    권장 사용법: BigQuery 파라미터를 전달하여 Tool이 데이터를 직접 조회하도록 함
    하위 호환성: timeseries_data와 news_articles를 직접 전달하는 것도 가능하지만 권장하지 않음
    """
    return orchestrate_analysis(
        timeseries_data=input_data.timeseries_data,
        news_articles=input_data.news_articles,
        context=input_data.context or "금융 시장 분석",
        timeseries_table_id=input_data.timeseries_table_id,
        timeseries_value_column=input_data.timeseries_value_column,
        timeseries_days=input_data.timeseries_days,
        news_table_id=input_data.news_table_id,
        news_value_column=input_data.news_value_column,
        news_days=input_data.news_days
    )


@router.get("/health")
async def health_check():
    """Orchestrator 상태 확인"""
    return {
        "status": "healthy",
        "components": {
            "timeseries": "ready",
            "sentiment": "ready",
            "llm": "ready"
        },
        "timestamp": datetime.now().isoformat()
    }