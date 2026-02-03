import json
from schema.models import OrchestratorOutput, TimeSeriesPrediction, SentimentAnalysis
from models.llm_summarizer import LLMSummarizer
from datetime import datetime
from typing import Optional, Tuple, Union, Dict, Any

# 모델 초기화 (Lazy initialization)
llm_summarizer = None


def get_llm_summarizer():
    """LLM Summarizer 지연 초기화"""
    global llm_summarizer
    if llm_summarizer is None:
        llm_summarizer = LLMSummarizer()
    return llm_summarizer


def parse_agent_result_raw(agent_result: dict) -> Dict[str, Any]:
    """
    Agent 실행 결과에서 DB 적재를 위한 Raw 데이터를 추출합니다.
    Airflow 파이프라인에서 사용됩니다.
    """
    from langchain_core.messages import ToolMessage

    messages = agent_result.get("messages", []) if isinstance(agent_result, dict) else []
    
    extracted_data = {
        "timeseries_data": None,
        "news_data": None
    }

    for msg in messages:
        if isinstance(msg, ToolMessage):
            # 시계열 예측 결과 (JSON)
            if msg.name == "timeseries_predictor":
                try:
                    ts_data = json.loads(msg.content)
                    if "error" not in ts_data:
                        extracted_data["timeseries_data"] = ts_data
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse timeseries JSON in raw extractor")

            # 뉴스 분석 결과 (JSON)
            elif msg.name == "news_sentiment_analyzer":
                try:
                    news_res = json.loads(msg.content)
                    if "error" not in news_res:
                        extracted_data["news_data"] = news_res
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse news analysis JSON in raw extractor")

    return extracted_data


def run_market_analysis(target_date: Optional[str] = None, context: str = "금융 시장 분석") -> Dict[str, Any]:
    """
    [Airflow용] 시장 분석을 실행하고 적재할 모든 데이터를 반환합니다.
    
    Returns:
        dict: {
            "target_date": str,
            "timeseries_data": dict, # BQ 적재용
            "news_data": dict,       # BQ 적재용
            "final_report": str      # GCS 적재용
        }
    """
    summarizer = get_llm_summarizer()

    # 분석 기준일이 없으면 오늘 날짜 사용
    if not target_date:
        target_date = datetime.now().strftime("%Y-%m-%d")

    print(f"🚀 Analyzing market for date: {target_date}")
    result = summarizer.summarize(context=context, target_date=target_date)
    
    # 1. Agent Tool 결과에서 데이터 추출
    agent_result = result.get("agent_result", {})
    raw_data = parse_agent_result_raw(agent_result)
    
    # 2. 최종 결과 조합
    output = {
        "target_date": target_date,
        "timeseries_data": raw_data.get("timeseries_data"),
        "news_data": raw_data.get("news_data"),
        "final_report": result.get("summary", "")
    }
    
    return output


def parse_agent_result(agent_result: dict) -> Tuple[TimeSeriesPrediction, list]:
    """
    Agent 실행 결과에서 Tool 메시지를 파싱하여 구조화된 데이터 반환
    """
    from langchain_core.messages import ToolMessage

    messages = agent_result.get("messages", []) if isinstance(agent_result, dict) else []

    timeseries_prediction = None
    sentiment_analysis = []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            # 시계열 예측 결과 파싱 (JSON)
            if msg.name == "timeseries_predictor":
                try:
                    ts_data = json.loads(msg.content)
                    if "error" not in ts_data:
                        timeseries_prediction = TimeSeriesPrediction(
                            prediction=ts_data.get("forecast_value", 0.0),
                            confidence=ts_data.get("confidence_score", 0.0) / 100,
                            timestamp=ts_data.get("target_date", datetime.now().isoformat()),
                        )
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse timeseries JSON: {msg.content[:50]}...")

            # 뉴스 분석 결과 파싱 (JSON)
            elif msg.name == "news_sentiment_analyzer":
                try:
                    news_res = json.loads(msg.content)
                    if "error" not in news_res:
                        # 근거 뉴스들을 SentimentAnalysis 형식으로 변환하여 추가
                        for news in news_res.get("evidence_news", []):
                            impact = news.get("price_impact_score", 0.0)
                            # 점수 정규화 및 할당
                            scores = {
                                "positive": max(0.0, impact) if impact > 0 else 0.1,
                                "negative": abs(impact) if impact < 0 else 0.1,
                                "neutral": 0.5,
                            }
                            sentiment_analysis.append(
                                SentimentAnalysis(
                                    text=f"[{news.get('title', '제목없음')}] {news.get('all_text', '')[:200]}...",
                                    sentiment="positive" if impact > 0 else ("negative" if impact < 0 else "neutral"),
                                    scores=scores,
                                )
                            )
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse news analysis JSON: {msg.content[:50]}...")

    # 기본값 설정
    if not timeseries_prediction:
        timeseries_prediction = TimeSeriesPrediction(
            prediction=0.0, confidence=0.0, timestamp=datetime.now().isoformat()
        )

    return timeseries_prediction, sentiment_analysis


def orchestrate_analysis(
    target_date: Optional[str] = None, context: str = "금융 시장 분석", return_agent_result: bool = False, **kwargs
) -> Union[OrchestratorOutput, Tuple[OrchestratorOutput, dict]]:
    """
    Orchestrator 분석 로직
    """
    summarizer = get_llm_summarizer()

    # 분석 기준일이 없으면 오늘 날짜 사용
    if not target_date:
        target_date = datetime.now().strftime("%Y-%m-%d")

    result = summarizer.summarize(context=context, target_date=target_date)

    agent_result = result.get("agent_result", {})
    timeseries_prediction, sentiment_analysis = parse_agent_result(agent_result)

    output = OrchestratorOutput(
        timeseries_prediction=timeseries_prediction,
        sentiment_analysis=sentiment_analysis,
        llm_summary=result.get("summary", ""),
    )

    if return_agent_result:
        return output, agent_result
    return output
