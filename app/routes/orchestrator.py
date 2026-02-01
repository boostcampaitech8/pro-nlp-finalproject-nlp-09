import json
from schema.models import OrchestratorOutput, TimeSeriesPrediction, SentimentAnalysis
from models.llm_summarizer import LLMSummarizer
from datetime import datetime
from typing import Optional, Tuple, Union

# 모델 초기화 (Lazy initialization)
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
    """
    from langchain_core.messages import ToolMessage

    messages = (
        agent_result.get("messages", []) if isinstance(agent_result, dict) else []
    )

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
                            timestamp=ts_data.get(
                                "target_date", datetime.now().isoformat()
                            ),
                        )
                except json.JSONDecodeError:
                    print(
                        f"Warning: Failed to parse timeseries JSON: {msg.content[:50]}..."
                    )

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
                                    sentiment="positive"
                                    if impact > 0
                                    else ("negative" if impact < 0 else "neutral"),
                                    scores=scores,
                                )
                            )
                except json.JSONDecodeError:
                    print(
                        f"Warning: Failed to parse news analysis JSON: {msg.content[:50]}..."
                    )

    # 기본값 설정
    if not timeseries_prediction:
        timeseries_prediction = TimeSeriesPrediction(
            prediction=0.0, confidence=0.0, timestamp=datetime.now().isoformat()
        )

    return timeseries_prediction, sentiment_analysis


def orchestrate_analysis(
    target_date: Optional[str] = None,
    context: str = "금융 시장 분석",
    return_agent_result: bool = False,
    **kwargs,
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
