import streamlit as st
import pandas as pd
from src.visualizer import draw_main_chart, draw_sentiment_chart, draw_contribution_chart
from src.analytics import calculate_metrics
from src.bq_manager import get_performance_data

# 1. 페이지 설정 및 테마 정의
st.set_page_config(page_title="Market Intel Pro", layout="wide", initial_sidebar_state="collapsed")
def apply_custom_style():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        
        :root {
            /* 배경: 너무 검지 않은 짙은 네이비 톤 */
            --bg-deep: #0a0e17; 
            --bg-card: rgba(17, 25, 40, 0.75); 
            --accent-primary: #00e5ff;
            --accent-success: #00c853;
            --accent-error: #ff1744;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --border-thin: rgba(255, 255, 255, 0.08);
        }

        /* 메인 배경: 아주 은은한 광원 효과 추가 */
        [data-testid="stAppViewContainer"] {
            background: 
                radial-gradient(circle at 10% 10%, rgba(0, 229, 255, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 90%, rgba(0, 200, 83, 0.03) 0%, transparent 40%),
                var(--bg-deep);
            font-family: 'Inter', sans-serif;
            color: var(--text-main);
        }

        /* 카드: 글래스모피즘 + 날카로운 테두리 */
        .custom-card {
            background: var(--bg-card);
            border: 1px solid var(--border-thin);
            border-radius: 6px; /* 약간의 곡률만 부여 */
            padding: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(8px);
            margin-bottom: 24px;
        }
        
        .custom-card:hover {
            border-color: rgba(0, 229, 255, 0.2);
            background: rgba(23, 32, 51, 0.85);
        }

        .card-header {
            font-size: 1.25rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: 0.02em;
            margin-bottom: 16px;
            opacity: 0.95;
        }

        /* Metric: 강조 + 호버 효과 */
        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 12px;
            padding: 14px 18px !important;
            box-shadow: 0 10px 24px rgba(2, 6, 23, 0.55);
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 36px rgba(2, 6, 23, 0.7);
            border-color: rgba(56, 189, 248, 0.6);
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.95rem !important;
            color: #cbd5e1 !important;
            font-weight: 700 !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 2.0rem !important;
            font-weight: 800 !important;
            color: #f8fafc !important;
            font-variant-numeric: tabular-nums;
        }

        /* 사이드바: 가독성 강화 */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%) !important;
            border-right: 1px solid var(--border-thin) !important;
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.06) !important;
            color: #f8fafc !important;
            border: 1px solid rgba(148, 163, 184, 0.35) !important;
            border-radius: 12px !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] > div:hover {
            border-color: rgba(56, 189, 248, 0.6) !important;
        }

        /* 제목: 강조 + 자연스러운 톤 */
        .page-title {
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: #e2e8f0;
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 0 0 6px 0;
        }
        .title-badge {
            width: 16px;
            height: 16px;
            border-radius: 4px;
            background: linear-gradient(135deg, #38bdf8 0%, #22d3ee 100%);
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.35);
            display: inline-block;
        }
        .title-sub {
            font-size: 1.05rem;
            color: #94a3b8;
            font-weight: 600;
        }
        [data-testid="stHeader"] {
            background: transparent !important;
        }

        /* 데이터프레임 가독성 */
        .stDataFrame {
            background: rgba(15, 23, 42, 0.3);
            border-radius: 4px;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_style()

# 카드 래퍼 함수
def card_begin(title):
    st.markdown(f'<div class="custom-card"><div class="card-header">{title}</div>', unsafe_allow_html=True)

def card_end():
    st.markdown('</div>', unsafe_allow_html=True)

# 2. 사이드바 - 정돈된 컨트롤러
with st.sidebar:
    st.markdown("### 🌾 종목 설정")
    commodity_label = st.radio(
        "종목 선택",
        ["옥수수", "밀", "대두"],
        key="commodity_label",
        label_visibility="collapsed",
        horizontal=True,
    )
    commodity_map = {"옥수수": "corn", "밀": "wheat", "대두": "soybean"}
    commodity = commodity_map[commodity_label]

    st.markdown("### ⏱️ 기간 설정")
    range_mode = st.selectbox(
        "분석 기간",
        ["최근 7일", "최근 30일", "YTD", "1년", "커스텀"],
        key="range_mode",
        label_visibility="collapsed",
    )

    today = pd.Timestamp.today().date()
    if range_mode == "최근 7일":
        start_date, end_date = today - pd.Timedelta(days=6), today
    elif range_mode == "최근 30일":
        start_date, end_date = today - pd.Timedelta(days=29), today
    elif range_mode == "YTD":
        start_date, end_date = pd.Timestamp(today.year, 1, 1).date(), today
    elif range_mode == "1년":
        start_date, end_date = today - pd.Timedelta(days=365), today
    else:
        start_date = st.date_input("시작일", today)
        end_date = st.date_input("종료일", today)

# 데이터 로드
filtered_df = get_performance_data(
    start_date=start_date,
    end_date=end_date,
    commodity=commodity,
)
if filtered_df.empty:
    st.warning("데이터가 존재하지 않습니다.")
    st.stop()

metrics = calculate_metrics(filtered_df)

# 타이틀 섹션
st.markdown(
    f"""
    <div class="page-title">
        <span class="title-badge"></span>
        시장 예측 성과 리포트
        <span class="title-sub">[{commodity_label}]</span>
        <span class="title-sub">{start_date} ~ {end_date}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# 3. KPI Metrics - 가로 배치 카드
st.markdown("### 📌 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🎯 방향 적중률", f"{metrics['hit_rate']}%", "Market Trend")
with col2:
    st.metric("📉 평균 오차 (MAE)", f"${metrics['mae']}", "Forecast Bias")
with col3:
    st.metric("⚠️ 리스크 (RMSE)", f"${metrics['rmse']}", "Volatility Risk", delta_color="inverse")
with col4:
    # 예시: 이전 기간 대비 변동성 등 추가 지표
    st.metric("📊 데이터 샘플", f"{len(filtered_df)}개", "Time-series")

st.markdown("---")

# 4. 메인 분석 차트 - 와이드 레이아웃
card_begin("📡 가격 예측 결과 및 신뢰 구간")
main_fig = draw_main_chart(filtered_df)
# Plotly 배경 투명화 처리 권장 (draw_main_chart 내부에서 실행)
st.plotly_chart(main_fig, use_container_width=True)
card_end()

# 5. 하단 상세 분석 - 전체 폭 (크기 확대)
card_begin("🧩 핵심 요인 기여도 & 뉴스 감성")
t1, t2 = st.tabs(["기여도 추이", "뉴스 감성"])
with t1:
    contrib_fig = draw_contribution_chart(filtered_df)
    st.plotly_chart(contrib_fig, use_container_width=True) if contrib_fig else st.info("기여도 데이터 없음")
with t2:
    if "sentiment" in filtered_df.columns:
        st.plotly_chart(draw_sentiment_chart(filtered_df), use_container_width=True)
card_end()

card_begin("📋 최근 예측 상세 로그")
# 테이블 가독성 향상을 위해 필요한 컬럼만 추출 및 포맷팅
display_df = filtered_df.sort_values("target_date", ascending=False).head(10).copy()
if "direction" in display_df.columns:
    display_df["Trend"] = display_df["direction"].apply(lambda x: "🟢 상승" if x == 1 else "🔴 하락")

st.dataframe(
    display_df[["target_date", "actual_price", "forecast_price", "Trend"]],
    use_container_width=True,
    hide_index=True
)
card_end()
