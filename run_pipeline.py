"""
금융 분석 파이프라인 - 직접 실행 스크립트
매일 1회 실행으로 시계열 예측 + 감성분석 + LLM 요약 수행
Vertex AI를 사용하는 LangChain Agent 방식
"""

import sys
import os
from datetime import datetime
from typing import List

# 프로젝트 경로 설정
project_root = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(project_root, 'app')
sys.path.insert(0, app_dir)

from routes.orchestrator import orchestrate_analysis
from utils.bigquery_client import BigQueryClient


# def get_sample_timeseries_data() -> List[float]:
#     """예시 시계열 데이터 생성 (실제 데이터로 대체 가능)"""
#     # 60일치 예시 주가 데이터 (상승 추세)
#     import random
#     base_price = 100.0
#     data = []
#     for i in range(60):
#         base_price += random.uniform(-2, 3)
#         data.append(round(base_price, 2))
#     return data


# 하위 호환성을 위한 함수들 (현재는 사용되지 않음)
# 현재는 Agent가 Tool을 통해 BigQuery에서 직접 데이터를 가져오므로 이 함수들은 사용되지 않습니다.
# 필요시 주석을 해제하여 사용할 수 있습니다.

def get_timeseries_data() -> List[float]:
    """BigQuery에서 corn_price 테이블의 close 컬럼 데이터 가져오기"""
    client = BigQueryClient()
    data = client.get_timeseries_data(
        table_id="corn_price",
        value_column="close",
        date_column="time",
        days=30
    )
    # close 컬럼 값만 추출하여 리스트로 변환
    values = []
    for item in data:
        value = item.get('close')
        if value is not None:
            values.append(float(value))
    return values


def get_news_texts() -> List[str]:
    """BigQuery에서 news_article 테이블의 description 컬럼 데이터 가져오기"""
    client = BigQueryClient()
    data = client.get_timeseries_data(
        table_id="news_article",
        value_column="description",
        date_column="publish_date",
        where_clause="filter_status = 'T'",
        days=3
    )
    # description 컬럼만 추출하여 리스트로 변환
    descriptions = []
    for item in data:
        desc = item.get('description')
        if desc:
            descriptions.append(str(desc))
    return descriptions


def main():
    """메인 파이프라인 실행"""
    
    print("=" * 70)
    print("금융 분석 파이프라인 시작 (Vertex AI + LangChain Agent)")
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # 1. BigQuery에서 데이터 가져오기
        print("\n[단계 1] BigQuery에서 데이터 가져오기...")
        
        print("   - 시계열 데이터 가져오는 중...")
        timeseries_data = get_timeseries_data()
        print(f"     → {len(timeseries_data)}개 데이터 포인트 가져옴")
        
        print("   - 뉴스 기사 가져오는 중...")
        news_articles = get_news_texts()
        print(f"     → {len(news_articles)}개 기사 가져옴")
        
        # 2. Orchestrator를 통한 분석 실행
        print("\n[단계 2] Orchestrator 분석 실행 중...")
        print("   Orchestrator가 다음 작업을 수행합니다:")
        print("   1. LangChain Agent 초기화")
        print("   2. 시계열 예측 Tool 호출")
        print("   3. 뉴스 감성분석 Tool 호출")
        print("   4. 결과를 바탕으로 통합 요약 생성")
        print("-" * 70)
        
        # Orchestrator 함수 직접 호출 (agent_result도 함께 받음)
        # 직접 데이터를 전달
        result, agent_result = orchestrate_analysis(
            context=f"일일 금융 시장 분석 ({datetime.now().strftime('%Y-%m-%d')})",
            timeseries_data=timeseries_data,
            news_articles=news_articles,
            return_agent_result=True
        )
        
        # 3. 결과 출력
        print("\n[단계 3] 분석 결과")
        print("=" * 70)
        print(result.llm_summary)
        print("=" * 70)
        
        # Tool 결과도 출력 (선택사항)
        print(f"\n[Tool 실행 결과]")
        print(f"  - 시계열 예측: {result.timeseries_prediction.prediction:.2f} (신뢰도: {result.timeseries_prediction.confidence:.2%})")
        print(f"  - 감성분석: {len(result.sentiment_analysis)}개 기사 분석 완료")
        
        # 4. 결과 저장 (summary와 agent 결과를 별도 파일로 저장)
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
        output_dir = os.path.join(project_root, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # 1. Summary 파일 저장
        summary_file = os.path.join(output_dir, f'summary_{timestamp}.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(result.llm_summary)
        
        print(f"\n💾 Summary 저장: {summary_file}")
        print(f"   - 길이: {len(result.llm_summary)}자")
        
        # 2. Agent 결과 전체 파일 저장
        agent_file = os.path.join(output_dir, f'agent_result_{timestamp}.txt')
        with open(agent_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n\n")
            f.write("Agent 실행 결과 전체\n")
            f.write("=" * 70 + "\n\n")
            
            # Agent 결과 구조화하여 저장
            from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
            import json
            
            messages = agent_result.get('messages', []) if isinstance(agent_result, dict) else []
            
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
                            f.write(f"  [{j}] Tool: {tool_call.get('name', 'unknown')}\n")
                            f.write(f"      Args: {json.dumps(tool_call.get('args', {}), ensure_ascii=False, indent=2)}\n")
                    if msg.content:
                        f.write(f"내용:\n{msg.content}\n")
                    f.write("\n")
                
                elif isinstance(msg, ToolMessage):
                    f.write(f"타입: ToolMessage\n")
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
                    f.write(f"  - 예측값: {result.timeseries_prediction.prediction:.2f}\n")
                    f.write(f"  - 신뢰도: {result.timeseries_prediction.confidence:.2%}\n")
                    f.write(f"  - 타임스탬프: {result.timeseries_prediction.timestamp}\n\n")
                
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
                serializable_result = {
                    'messages': []
                }
                for msg in messages:
                    msg_dict = {
                        'type': type(msg).__name__,
                        'content': str(msg.content) if hasattr(msg, 'content') else str(msg)
                    }
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        msg_dict['tool_calls'] = [
                            {
                                'name': tc.get('name', 'unknown'),
                                'args': tc.get('args', {})
                            }
                            for tc in msg.tool_calls
                        ]
                    if isinstance(msg, ToolMessage):
                        msg_dict['name'] = msg.name
                    serializable_result['messages'].append(msg_dict)
                
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