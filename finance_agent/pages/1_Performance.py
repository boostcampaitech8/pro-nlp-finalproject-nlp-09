import streamlit as st
import pandas as pd
from src.visualizer import (
    draw_main_chart,
    draw_sentiment_chart,
    draw_contribution_chart,
    draw_volume_volatility_bubble,
)
from src.analytics import calculate_metrics
from src.bq_manager import get_performance_data, get_sentiment_result_data

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
        [data-baseweb="tab-list"] button[role="tab"] {
            color: #cbd5e1 !important;
            font-weight: 700 !important;
        }
        [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
            color: #f8fafc !important;
        }
        [data-testid="stMarkdownContainer"] h4 {
            color: #e2e8f0 !important;
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
        .commodity-tag {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(56, 189, 248, 0.45);
            background: rgba(14, 165, 233, 0.12);
            font-size: 0.9rem;
            font-weight: 700;
            color: #e2e8f0;
            margin-left: 8px;
        }
        .commodity-icon {
            width: 16px;
            height: 16px;
            display: inline-block;
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
        index=3,
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

# 뉴스 감성 데이터 로드 및 조인
sentiment_df = get_sentiment_result_data(
    start_date=start_date,
    end_date=end_date,
    keyword=commodity,
)
if not sentiment_df.empty:
    sentiment_df = sentiment_df.copy()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"]).dt.date
    sentiment_df["target_date"] = sentiment_df["date"]
    sentiment_df["news_count"] = pd.to_numeric(sentiment_df["news_count"], errors="coerce").fillna(0)
    sentiment_df["pos_ratio"] = pd.to_numeric(sentiment_df["pos_ratio"], errors="coerce").fillna(0)
    sentiment_df["neg_ratio"] = pd.to_numeric(sentiment_df["neg_ratio"], errors="coerce").fillna(0)
    # 날짜별 중복 레코드가 있을 수 있어 집계 후 조인
    sentiment_df = (
        sentiment_df.groupby("target_date", as_index=False)
        .agg(
            news_count=("news_count", "sum"),
            pos_ratio=("pos_ratio", "mean"),
            neg_ratio=("neg_ratio", "mean"),
        )
    )
    sentiment_df["sentiment"] = (sentiment_df["pos_ratio"] - sentiment_df["neg_ratio"]) * 100

    filtered_df["target_date"] = pd.to_datetime(filtered_df["target_date"]).dt.date
    filtered_df = filtered_df.merge(
        sentiment_df[["target_date", "news_count", "pos_ratio", "neg_ratio", "sentiment"]],
        on="target_date",
        how="left",
    )

metrics = calculate_metrics(filtered_df)

# 뉴스 감성 대비 다음날 가격 반응 지표
news_alignment_ratio = None
pos_up_ratio = None
pos_down_ratio = None
if {"target_date", "sentiment", "actual_price"}.issubset(filtered_df.columns):
    rel_df = filtered_df.copy()
    rel_df["target_date"] = pd.to_datetime(rel_df["target_date"])
    rel_df["sentiment"] = pd.to_numeric(rel_df["sentiment"], errors="coerce")
    rel_df["actual_price"] = pd.to_numeric(rel_df["actual_price"], errors="coerce")
    rel_df = rel_df.dropna(subset=["target_date", "sentiment", "actual_price"]).sort_values("target_date")

    if not rel_df.empty:
        rel_df = (
            rel_df.groupby("target_date", as_index=False)
            .agg(sentiment=("sentiment", "mean"), actual_price=("actual_price", "mean"))
        )
        rel_df["next_ret"] = rel_df["actual_price"].shift(-1) - rel_df["actual_price"]

        valid = rel_df[(rel_df["sentiment"] != 0) & rel_df["next_ret"].notna()].copy()
        if not valid.empty:
            valid["sent_sign"] = valid["sentiment"].apply(lambda x: 1 if x > 0 else -1)
            valid["ret_sign"] = valid["next_ret"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            match = valid[valid["ret_sign"] != 0]
            if not match.empty:
                news_alignment_ratio = round(float((match["sent_sign"] == match["ret_sign"]).mean() * 100), 1)

        pos_only = rel_df[(rel_df["sentiment"] > 0) & rel_df["next_ret"].notna()].copy()
        if not pos_only.empty:
            pos_up_ratio = round(float((pos_only["next_ret"] > 0).mean() * 100), 1)
            pos_down_ratio = round(float((pos_only["next_ret"] < 0).mean() * 100), 1)

# 아이콘/라벨
commodity_icon_map = {
    "corn": (
        '<path d="M8 1.2 C6.5 1.2 4.8 3.5 4.8 7.5 C4.8 11.5 6.2 14 8 14 C9.8 14 11.2 11.5 11.2 7.5 C11.2 3.5 9.5 1.2 8 1.2 Z" fill="#FFD750" stroke="#B8860B" stroke-width="0.25" />'
        '<path d="M5.2 4.5 Q8 4.2 10.8 4.5 M4.9 7 Q8 6.7 11.1 7 M5 9.5 Q8 9.2 11 9.5 M5.5 12 Q8 11.7 10.5 12" fill="none" stroke="#B8860B" stroke-width="0.25" opacity="0.5" />'
        '<path d="M6.5 2 Q6.5 7.5 6.5 13.5 M8 1.2 Q8 7.5 8 14 M9.5 2 Q9.5 7.5 9.5 13.5" fill="none" stroke="#B8860B" stroke-width="0.25" opacity="0.5" />'
        '<path d="M8 14 C5 14 1.5 11 2 6.5 C2.2 4 4.5 3 5.5 4.5 C4.5 7 4.5 11 8 14 Z" fill="#4CAF50" stroke="#2E7D32" stroke-width="0.3" />'
        '<path d="M8 14 C11 14 14.5 11 14 6.5 C13.8 4 11.5 3 10.5 4.5 C11.5 7 11.5 11 8 14 Z" fill="#4CAF50" stroke="#2E7D32" stroke-width="0.3" />'
        '<path d="M8 14 C6.2 14 5 12 5 8.5 C6 9.5 7 10.5 8 14 Z" fill="#66BB6A" stroke="#2E7D32" stroke-width="0.25" />'
        '<path d="M8 14 C9.8 14 11 12 11 8.5 C10 9.5 9 10.5 8 14 Z" fill="#66BB6A" stroke="#2E7D32" stroke-width="0.25" />'
        '<path d="M7.6 14 L7.6 15.8 C7.6 16.2 8.4 16.2 8.4 15.8 L8.4 14" fill="#4CAF50" stroke="#2E7D32" stroke-width="0.3" />'
    ),
    "wheat": (
        '<path d="M8 15 L8 2" stroke="#8B4513" stroke-width="0.3" fill="none" />'
        '<path d="M8 14 C6 14 4.5 13 4.5 11.5 C6 11.5 7.5 12.5 8 14 Z" fill="#66BB6A" stroke="#2E7D32" stroke-width="0.25" />'
        '<path d="M8 14 C10 14 11.5 13 11.5 11.5 C10 11.5 8.5 12.5 8 14 Z" fill="#66BB6A" stroke="#2E7D32" stroke-width="0.25" />'
        '<path d="M8 11.5 C6.5 11.5 5.5 10 5.5 8.5 C7 8.5 8 10 8 11.5 Z" fill="#F0E68C" stroke="#B8860B" stroke-width="0.25" />'
        '<path d="M8 11.5 C9.5 11.5 10.5 10 10.5 8.5 C9 8.5 8 10 8 11.5 Z" fill="#EBC050" stroke="#B8860B" stroke-width="0.25" />'
        '<path d="M8 9 C6.5 9 5.5 7.5 5.5 6 C7 6 8 7.5 8 9 Z" fill="#F0E68C" stroke="#B8860B" stroke-width="0.25" />'
        '<path d="M8 9 C9.5 9 10.5 7.5 10.5 6 C9 6 8 7.5 8 9 Z" fill="#EBC050" stroke="#B8860B" stroke-width="0.25" />'
        '<path d="M8 6.5 C6.8 6.5 6 5.5 6 4.5 C7 4.5 8 5.5 8 6.5 Z" fill="#F0E68C" stroke="#B8860B" stroke-width="0.25" />'
        '<path d="M8 6.5 C9.2 6.5 10 5.5 10 4.5 C9 4.5 8 5.5 8 6.5 Z" fill="#EBC050" stroke="#B8860B" stroke-width="0.25" />'
        '<path d="M8 4.5 C7.5 4.5 7.5 3 8 2 C8.5 3 8.5 4.5 8 4.5 Z" fill="#F0E68C" stroke="#B8860B" stroke-width="0.25" />'
    ),
    "soybean": (
        '<path d="M4 8.2 C4 4.5 7 3 10 3 C13 3 14 6 14 8.5 C14 12 11 14 8 14 C5 14 4 11.5 4 8.2 Z" fill="#8DB600" />'
        '<circle cx="8" cy="6.5" r="1.8" fill="#B2D300" />'
        '<circle cx="9" cy="10.5" r="1.8" fill="#B2D300" />'
        '<path d="M6.5 5.5 C7 5 8 4.8 9 5" stroke="white" stroke-width="0.5" fill="none" opacity="0.6" />'
    ),
}
commodity_label_map = {"corn": "옥수수", "wheat": "밀", "soybean": "대두"}
commodity_label_upper = commodity_label_map.get(commodity, commodity.upper())
commodity_svg = commodity_icon_map.get(commodity, commodity_icon_map["corn"])

# 타이틀 섹션
st.markdown(
    f"""
    <div class="page-title">
        <span class="title-badge"></span>
        시장 예측 성과 리포트
        <span class="commodity-tag">
            <svg class="commodity-icon" viewBox="0 0 16 16">
                {commodity_svg}
            </svg>
            {commodity_label_upper}
        </span>
        <span class="title-sub">{start_date} ~ {end_date}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# 3. KPI Metrics - 가로 배치 카드
st.markdown("### 📌 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("🎯 방향 적중률", f"{metrics['hit_rate']}%", "Market Trend")
with col2:
    st.metric("📉 평균 오차 (MAE)", f"${metrics['mae']}", "Forecast Bias")
with col3:
    st.metric("⚠️ 리스크 (RMSE)", f"${metrics['rmse']}", "Volatility Risk", delta_color="inverse")
with col4:
    st.metric("📊 데이터 샘플", f"{len(filtered_df)}개", "Time-series")
with col5:
    if news_alignment_ratio is None:
        st.metric("📰 뉴스-가격 일치율", "N/A", "Sentiment vs Next day")
    else:
        st.metric("📰 뉴스-가격 일치율", f"{news_alignment_ratio}%", "Sentiment vs Next day")

if pos_up_ratio is not None and pos_down_ratio is not None:
    st.caption(f"긍정 뉴스 후 다음날: 상승 {pos_up_ratio}% · 하락 {pos_down_ratio}%")

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
    if contrib_fig:
        st.plotly_chart(contrib_fig, use_container_width=True)
    else:
        st.info("기여도 데이터 없음")
with t2:
    if "sentiment" in filtered_df.columns:
        sentiment_fig = draw_sentiment_chart(filtered_df)
        bubble_fig = draw_volume_volatility_bubble(filtered_df)

        if sentiment_fig:
            st.plotly_chart(sentiment_fig, use_container_width=True)
        if bubble_fig:
            st.markdown("#### 2) 정보 집중도와 가격 변동성")
            st.plotly_chart(bubble_fig, use_container_width=True)
        if not sentiment_fig and not bubble_fig:
            st.info("뉴스 감성 시각화용 데이터가 충분하지 않습니다.")
    else:
        st.info("뉴스 감성 데이터가 없습니다.")
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
