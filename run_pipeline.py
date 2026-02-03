"""
금융 분석 파이프라인 - 직접 실행 스크립트
매일 1회 실행으로 시계열 예측 + 감성분석 + LLM 요약 수행
Vertex AI를 사용하는 LangChain Agent 방식 (날짜 기반 자동 조회)
"""

import logging
import sys
import os
from datetime import datetime
from app.routes.orchestrator import orchestrate_analysis

# 프로젝트 경로 설정
_project_root = os.path.dirname(os.path.abspath(__file__))


def setup_logging():
    # 1. 환경 변수에서 로그 레벨 가져오기 (기본값: INFO)
    # 터미널에서 LOG_LEVEL=DEBUG 라고 치면 DEBUG로 변함
    log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()

    # 2. "루트 로거(Root Logger)" 설정 (이게 핵심!)
    # 여기서 설정하면 logging.getLogger(__name__)을 쓴 모든 모듈에 전파됨
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] [%(name)s] %(message)s",
        stream=sys.stdout,  # 혹은 sys.stderr
    )

    noisy_loggers = [
        "openai",
        "httpx",
        "httpcore",
        "urllib3",
        "google",
        "google.auth",
        "google.api_core",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    logging.getLogger("httpcore").propagate = False


def main():
    setup_logging()
    """메인 파이프라인 실행"""

    # 분석 기준 날짜 설정 (기본값: 오늘, 또는 테스트용 특정 날짜)
    # 실제 운영시에는 datetime.now().strftime('%Y-%m-%d') 사용
    target_date = "2025-11-14"
    current_commodity = "corn"

    print("=" * 70)
    print("금융 분석 파이프라인 시작 (Vertex AI + LangChain Agent)")
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"분석 기준일: {target_date}")
    print("=" * 70)

    try:
        # 1. Orchestrator를 통한 분석 실행
        print(f"\n[단계 1] Orchestrator 분석 실행 중 ({target_date})...")
        print("   Orchestrator가 다음 작업을 수행합니다:")
        print("   1. LangChain Agent 초기화")
        print("   2. Agent가 날짜를 기반으로 도구(Tool) 호출")
        print("      - timeseries_predictor: BigQuery 피처 조회 -> XGBoost 예측")
        print("      - news_sentiment_analyzer: BigQuery 뉴스 조회 -> 시장 영향력 예측")
        print("   3. 결과를 바탕으로 통합 요약 생성")
        print("-" * 70)

        # Orchestrator 함수 직접 호출
        result, agent_result = orchestrate_analysis(
            target_date=target_date,
            commodity=current_commodity,
            context=f"일일 금융 시장 분석 ({target_date})",
            return_agent_result=True,
        )

        # 2. 결과 출력
        print("\n[단계 2] 분석 결과")
        print("=" * 70)
        print(result.llm_summary)
        print("=" * 70)

        # Tool 결과 요약
        print("\n[Tool 실행 결과 요약]")
        if result.timeseries_prediction:
            print(
                f"  - 시계열 예측: {result.timeseries_prediction.prediction:.2f} (신뢰도: {result.timeseries_prediction.confidence:.2%})"
            )
        if result.sentiment_analysis:
            print(f"  - 근거 뉴스: {len(result.sentiment_analysis)}건 추출됨")
            for i, news in enumerate(result.sentiment_analysis[:3], 1):
                print(f"    {i}. [{news.sentiment}] {news.text[:50]}...")

        # 3. 결과 저장
        save_results_from_orchestrator(result, agent_result)

        print("\n✅ 파이프라인 완료!")
        print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        return 1


def save_results_from_orchestrator(result, agent_result: dict):
    """
    Orchestrator 결과를 별도 파일로 저장
    - summary: LLM 요약만 저장
    - agent_result: Agent 실행 결과 전체 저장

    Args:
        result: OrchestratorOutput 객체
        agent_result: Agent 실행 결과 전체 (Tool 메시지 포함)
    """
    try:
        # 결과 저장 디렉토리 생성
        output_dir = os.path.join(_project_root, "outputs")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # 1. Summary 파일 저장
        summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(result.llm_summary)

        print(f"\n💾 Summary 저장: {summary_file}")
        print(f"   - 길이: {len(result.llm_summary)}자")

        # 2. Agent 결과 전체 파일 저장
        agent_file = os.path.join(output_dir, f"agent_result_{timestamp}.txt")
        with open(agent_file, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n\n")
            f.write("Agent 실행 결과 전체\n")
            f.write("=" * 70 + "\n\n")

            # Agent 결과 구조화하여 저장
            from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
            import json

            messages = (
                agent_result.get("messages", [])
                if isinstance(agent_result, dict)
                else []
            )

            f.write(f"총 메시지 수: {len(messages)}\n\n")

            for i, msg in enumerate(messages, 1):
                f.write("-" * 70 + "\n")
                f.write(f"[메시지 {i}]\n")
                f.write("-" * 70 + "\n")

                if isinstance(msg, HumanMessage):
                    f.write("타입: HumanMessage\n")
                    f.write(f"내용:\n{msg.content}\n\n")

                elif isinstance(msg, AIMessage):
                    f.write("타입: AIMessage\n")
                    if msg.tool_calls:
                        f.write(f"Tool 호출 수: {len(msg.tool_calls)}\n")
                        for j, tool_call in enumerate(msg.tool_calls, 1):
                            f.write(
                                f"  [{j}] Tool: {tool_call.get('name', 'unknown')}\n"
                            )
                            f.write(
                                f"      Args: {json.dumps(tool_call.get('args', {}), ensure_ascii=False, indent=2)}\n"
                            )
                    if msg.content:
                        f.write(f"내용:\n{msg.content}\n")
                    f.write("\n")

                elif isinstance(msg, ToolMessage):
                    f.write("타입: ToolMessage\n")
                    f.write(f"Tool 이름: {msg.name}\n")
                    f.write(f"결과:\n{msg.content}\n\n")

                else:
                    f.write(f"타입: {type(msg).__name__}\n")
                    f.write(f"내용: {str(msg)}\n\n")

            # Tool 실행 결과 요약
            if result.timeseries_prediction or result.sentiment_analysis:
                f.write("\n" + "=" * 70 + "\n")
                f.write("파싱된 Tool 실행 결과\n")
                f.write("=" * 70 + "\n\n")

                if result.timeseries_prediction:
                    f.write("시계열 예측:\n")
                    f.write(
                        f"  - 예측값: {result.timeseries_prediction.prediction:.2f}\n"
                    )
                    f.write(
                        f"  - 신뢰도: {result.timeseries_prediction.confidence:.2%}\n"
                    )
                    f.write(
                        f"  - 타임스탬프: {result.timeseries_prediction.timestamp}\n\n"
                    )

                if result.sentiment_analysis:
                    f.write(f"감성분석 ({len(result.sentiment_analysis)}개 기사):\n")
                    for i, sa in enumerate(result.sentiment_analysis, 1):
                        f.write(f"  [{i}] {sa.sentiment}: {sa.text[:100]}...\n")
                    f.write("\n")

            # Agent 결과 원본 (JSON 형식, 선택사항)
            f.write("\n" + "=" * 70 + "\n")
            f.write("Agent 결과 원본 (JSON)\n")
            f.write("=" * 70 + "\n\n")
            try:
                # 메시지를 직렬화 가능한 형태로 변환
                serializable_result = {"messages": []}
                for msg in messages:
                    msg_dict = {
                        "type": type(msg).__name__,
                        "content": str(msg.content)
                        if hasattr(msg, "content")
                        else str(msg),
                    }
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        msg_dict["tool_calls"] = [
                            {
                                "name": tc.get("name", "unknown"),
                                "args": tc.get("args", {}),
                            }
                            for tc in msg.tool_calls
                        ]
                    if isinstance(msg, ToolMessage):
                        msg_dict["name"] = msg.name
                    serializable_result["messages"].append(msg_dict)

                f.write(json.dumps(serializable_result, ensure_ascii=False, indent=2))
            except Exception as e:
                f.write(f"JSON 직렬화 실패: {e}\n")
                f.write(f"원본: {str(agent_result)[:1000]}...\n")

        print(f"💾 Agent 결과 저장: {agent_file}")
        print(f"   - 메시지 수: {len(messages)}개")

    except Exception as e:
        print(f"\n⚠️  결과 저장 실패: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
