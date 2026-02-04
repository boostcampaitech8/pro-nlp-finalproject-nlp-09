import json
import logging
from app.schema.models import (
    OrchestratorOutput,
    TimeSeriesPrediction,
    SentimentAnalysis,
)
from app.models.llm_summarizer import LLMSummarizer
from datetime import datetime
from typing import Optional, Tuple, Union, Dict, Any

# 로거 설정
logger = logging.getLogger(__name__)

# 모델 초기화 (Lazy initialization)
llm_summarizer = None


def get_llm_summarizer():
    """LLM Summarizer 지연 초기화"""
    global llm_summarizer
    if llm_summarizer is None:
        logger.info("LLM Summarizer 초기화 시작 (지연 초기화)")
        llm_summarizer = LLMSummarizer()
        logger.info("LLM Summarizer 초기화 완료")
    else:
        logger.debug("기존 LLM Summarizer 인스턴스 재사용")
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

    logger.info("=" * 80)
    logger.info("parse_agent_result 시작")
    logger.debug(f"Agent result 타입: {type(agent_result)}")

    messages = (
        agent_result.get("messages", []) if isinstance(agent_result, dict) else []
    )

    logger.info(f"파싱할 메시지 총 개수: {len(messages)}")
    logger.debug(f"메시지 타입 목록: {[type(msg).__name__ for msg in messages]}")

    timeseries_prediction = None
    sentiment_analysis = []

    tool_message_count = 0
    for idx, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            tool_message_count += 1
            logger.info(
                f"ToolMessage 처리 중 #{tool_message_count} (인덱스 {idx}): {msg.name}"
            )
            logger.debug(f"도구 콘텐츠 길이: {len(msg.content)} 문자")
            logger.debug(f"도구 콘텐츠 미리보기: {msg.content[:200]}...")

            # 시계열 예측 결과 파싱 (JSON)
            if msg.name == "timeseries_predictor":
                logger.info("시계열 예측 결과 파싱 중")
                try:
                    ts_data = json.loads(msg.content)
                    logger.debug(f"파싱된 시계열 데이터 키: {list(ts_data.keys())}")

                    if "error" not in ts_data:
                        forecast_value = ts_data.get("forecast_value", 0.0)
                        confidence_score = ts_data.get("confidence_score", 0.0)
                        target_date = ts_data.get(
                            "target_date", datetime.now().isoformat()
                        )

                        logger.info(
                            f"시계열 예측 추출 완료: 예측값={forecast_value:.2f}, 신뢰도={confidence_score:.2f}%, 날짜={target_date}"
                        )

                        timeseries_prediction = TimeSeriesPrediction(
                            prediction=forecast_value,
                            confidence=confidence_score / 100,
                            timestamp=target_date,
                        )
                        logger.info("TimeSeriesPrediction 객체 생성 완료")
                    else:
                        error_msg = ts_data.get("error", "알 수 없는 오류")
                        logger.error(f"시계열 데이터에 오류 포함: {error_msg}")

                except json.JSONDecodeError as e:
                    logger.error(f"시계열 JSON 파싱 실패: {e}")
                    logger.error(f"유효하지 않은 JSON 콘텐츠: {msg.content[:200]}...")
                except Exception as e:
                    logger.error(
                        f"시계열 데이터 파싱 중 예상치 못한 오류: {e}", exc_info=True
                    )

            # 뉴스 분석 결과 파싱 (JSON)
            elif msg.name == "news_sentiment_analyzer":
                logger.info("뉴스 감성 분석 결과 파싱 중")
                try:
                    news_res = json.loads(msg.content)
                    logger.debug(f"파싱된 뉴스 결과 키: {list(news_res.keys())}")

                    if "error" not in news_res:
                        evidence_news = news_res.get("evidence_news", [])
                        logger.info(f"근거 뉴스 {len(evidence_news)}개 발견")

                        # 근거 뉴스들을 SentimentAnalysis 형식으로 변환하여 추가
                        for news_idx, news in enumerate(evidence_news):
                            impact = news.get("price_impact_score", 0.0)
                            title = news.get("title", "제목없음")
                            all_text = news.get("all_text", "")

                            logger.debug(
                                f"뉴스 #{news_idx + 1}: 제목='{title[:50]}...', 영향도={impact:.4f}"
                            )

                            # 점수 정규화 및 할당
                            scores = {
                                "positive": max(0.0, impact) if impact > 0 else 0.1,
                                "negative": abs(impact) if impact < 0 else 0.1,
                                "neutral": 0.5,
                            }

                            sentiment_label = (
                                "positive"
                                if impact > 0
                                else ("negative" if impact < 0 else "neutral")
                            )
                            logger.debug(
                                f"뉴스 #{news_idx + 1} 감성: {sentiment_label}, 점수={scores}"
                            )

                            sentiment_analysis.append(
                                SentimentAnalysis(
                                    text=f"[{title}] {all_text[:200]}...",
                                    sentiment=sentiment_label,
                                    scores=scores,
                                )
                            )

                        logger.info(
                            f"SentimentAnalysis 객체 {len(sentiment_analysis)}개 생성 완료"
                        )
                    else:
                        error_msg = news_res.get("error", "알 수 없는 오류")
                        logger.error(f"뉴스 분석 데이터에 오류 포함: {error_msg}")

                except json.JSONDecodeError as e:
                    logger.error(f"뉴스 분석 JSON 파싱 실패: {e}")
                    logger.error(f"유효하지 않은 JSON 콘텐츠: {msg.content[:200]}...")
                except Exception as e:
                    logger.error(
                        f"뉴스 분석 데이터 파싱 중 예상치 못한 오류: {e}", exc_info=True
                    )
            else:
                logger.debug(f"다음 이름의 ToolMessage 건너뛰기: {msg.name}")
        else:
            logger.debug(
                f"인덱스 {idx}의 비-ToolMessage 건너뛰기: {type(msg).__name__}"
            )

    logger.info(f"ToolMessage {tool_message_count}개 처리 완료")

    # 기본값 설정
    if not timeseries_prediction:
        logger.error("시계열 예측을 찾지 못함, 기본값 사용")
        timeseries_prediction = TimeSeriesPrediction(
            prediction=0.0, confidence=0.0, timestamp=datetime.now().isoformat()
        )

    logger.info(
        f"최종 결과: 시계열예측={timeseries_prediction is not None}, 감성분석개수={len(sentiment_analysis)}"
    )
    logger.info("parse_agent_result 완료")
    logger.info("=" * 80)

    return timeseries_prediction, sentiment_analysis


def orchestrate_analysis(
    target_date: Optional[str] = None,
    commodity: str = "corn",
    context: str = "금융 시장 분석",
    return_agent_result: bool = False,
    **kwargs,
) -> Union[OrchestratorOutput, Tuple[OrchestratorOutput, dict]]:
    """
    Orchestrator 분석 로직
    """
    logger.info("=" * 80)
    logger.info("orchestrate_analysis 시작")
    logger.info(
        f"파라미터: target_date={target_date}, commodity={commodity}, context='{context}'"
    )
    logger.info(f"옵션: return_agent_result={return_agent_result}")
    if kwargs:
        logger.debug(f"추가 kwargs: {kwargs}")

    try:
        summarizer = get_llm_summarizer()

        # 분석 기준일이 없으면 오늘 날짜 사용
        if not target_date:
            target_date = datetime.now().strftime("%Y-%m-%d")
            logger.info(f"target_date 미제공, 오늘 날짜 사용: {target_date}")
        else:
            logger.info(f"제공된 target_date 사용: {target_date}")

        logger.info(
            f"summarizer.summarize 호출: context='{context}', target_date={target_date}, commodity={commodity}"
        )
        result = summarizer.summarize(
            context=context, target_date=target_date, commodity=commodity
        )

        logger.info("Summarizer 실행 완료")
        logger.debug(
            f"결과 키: {list(result.keys()) if isinstance(result, dict) else '딕셔너리 아님'}"
        )

        agent_result = result.get("agent_result", {})
        logger.info(
            f"agent_result 추출: 타입={type(agent_result)}, messages_존재={'messages' in agent_result if isinstance(agent_result, dict) else 'N/A'}"
        )

        logger.info("agent result 파싱 중...")
        timeseries_prediction, sentiment_analysis = parse_agent_result(agent_result)

        logger.info("OrchestratorOutput 생성 중...")
        llm_summary = result.get("summary", "")
        logger.debug(f"LLM 요약 길이: {len(llm_summary)} 문자")
        logger.debug(
            f"LLM 요약 미리보기: {llm_summary[:200]}..." if llm_summary else "빈 요약"
        )

        output = OrchestratorOutput(
            timeseries_prediction=timeseries_prediction,
            sentiment_analysis=sentiment_analysis,
            llm_summary=llm_summary,
        )

        logger.info("OrchestratorOutput 생성 완료")
        logger.info(
            f"출력 요약: 시계열예측값={output.timeseries_prediction.prediction:.2f}, 감성분석개수={len(output.sentiment_analysis)}, 요약길이={len(output.llm_summary)}"
        )

        if return_agent_result:
            logger.info("agent_result와 함께 output 반환")
            logger.info("orchestrate_analysis 완료")
            logger.info("=" * 80)
            return output, agent_result
        else:
            logger.info("output만 반환")
            logger.info("orchestrate_analysis 완료")
            logger.info("=" * 80)
            return output

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"orchestrate_analysis에서 치명적 오류 발생: {e}", exc_info=True)
        logger.error(
            f"실패 시점의 파라미터: target_date={target_date}, commodity={commodity}, context='{context}'"
        )
        logger.error("=" * 80)
        raise
