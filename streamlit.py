"""
Summary 파일 뷰어 - Streamlit 앱
outputs 디렉토리에 저장된 summary_*.txt 파일들을 확인할 수 있습니다.
"""

import streamlit as st
import os
from pathlib import Path
from datetime import datetime
import glob

# 페이지 설정
st.set_page_config(
    page_title="Summary 뷰어",
    page_icon="📊",
    layout="wide"
)

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent
outputs_dir = project_root / "outputs"


def get_summary_files():
    """outputs 디렉토리에서 summary_*.txt 파일 목록 가져오기"""
    if not outputs_dir.exists():
        return []
    
    pattern = str(outputs_dir / "summary_*.txt")
    files = sorted(glob.glob(pattern), reverse=True)  # 최신 파일 먼저
    return files


def get_file_info(file_path):
    """파일 정보 가져오기"""
    stat = os.stat(file_path)
    return {
        "name": os.path.basename(file_path),
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime),
        "path": file_path
    }


def read_summary_file(file_path):
    """Summary 파일 읽기"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"파일 읽기 오류: {e}"


def get_agent_result_file(summary_file_path):
    """Summary 파일명에서 대응하는 agent_result 파일 찾기"""
    summary_name = os.path.basename(summary_file_path)
    # summary_2026-01-26_15-12-49.txt -> agent_result_2026-01-26_15-12-49.txt
    if summary_name.startswith("summary_"):
        agent_name = summary_name.replace("summary_", "agent_result_", 1)
        agent_path = outputs_dir / agent_name
        if agent_path.exists():
            return str(agent_path)
    return None


def read_agent_result_file(file_path):
    """Agent result 파일 읽기"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"파일 읽기 오류: {e}"


# 메인 UI
#st.title("📊 Summary 파일 뷰어")
#st.markdown("---")

# 파일 목록 가져오기
summary_files = get_summary_files()

if not summary_files:
    st.warning("⚠️ outputs 디렉토리에 summary 파일이 없습니다.")
    st.info("`run_pipeline.py`를 실행하여 summary 파일을 생성하세요.")
else:
    # 사이드바에 파일 선택
    st.sidebar.header("📁 파일 선택")
    
    # 파일 정보 리스트 생성
    file_info_list = [get_file_info(f) for f in summary_files]
    file_names = [f"{info['name']} ({info['modified'].strftime('%Y-%m-%d %H:%M:%S')})" 
                  for info in file_info_list]
    
    # 파일 선택
    selected_index = st.sidebar.selectbox(
        "파일을 선택하세요:",
        range(len(file_names)),
        format_func=lambda x: file_names[x]
    )
    
    selected_file = summary_files[selected_index]
    selected_info = file_info_list[selected_index]
    
    # 사이드바에 파일 정보 표시
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 파일 정보")
    st.sidebar.write(f"**파일명:** {selected_info['name']}")
    st.sidebar.write(f"**수정 시간:** {selected_info['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.write(f"**파일 크기:** {selected_info['size']:,} bytes")
    
    # 파일 개수 표시
    st.sidebar.markdown("---")
    st.sidebar.write(f"**총 파일 수:** {len(summary_files)}개")
    
    # 메인 영역에 파일 내용 표시
    st.subheader(f"📄 {selected_info['name']}")
    
    # 파일 내용 읽기
    content = read_summary_file(selected_file)
    
    # 대응하는 agent_result 파일 찾기
    agent_result_file = get_agent_result_file(selected_file)
    agent_result_content = None
    if agent_result_file:
        agent_result_content = read_agent_result_file(agent_result_file)
    
    # 탭으로 나누기
    if agent_result_content:
        tab1, tab2, tab3 = st.tabs(["📝 마크다운 보기", "📄 원본 텍스트", "🤖 Agent 결과"])
    else:
        tab1, tab2 = st.tabs(["📝 마크다운 보기", "📄 원본 텍스트"])
    
    with tab1:
        st.markdown(content)
    
    with tab2:
        st.code(content, language="text")
    
    if agent_result_content:
        with tab3:
            st.subheader("Agent 실행 결과")
            st.code(agent_result_content, language="text")
            
            # Agent 결과 다운로드 버튼
            agent_result_name = os.path.basename(agent_result_file)
            st.download_button(
                label="📥 Agent 결과 다운로드",
                data=agent_result_content,
                file_name=agent_result_name,
                mime="text/plain",
                key="agent_download"
            )
    
    # 다운로드 버튼
    st.download_button(
        label="📥 Summary 파일 다운로드",
        data=content,
        file_name=selected_info['name'],
        mime="text/plain"
    )
