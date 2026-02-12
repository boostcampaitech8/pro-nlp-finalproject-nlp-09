import streamlit as st
from src.analytics import calculate_backtest_comparison
from src.visualizer import draw_backtest_charts
from src.bq_manager import get_test_commodity_data, get_performance_data
import pandas as pd
import numpy as np

def load_mock_data():
    end = pd.Timestamp.today().normalize()
    dates = pd.date_range(end=end, periods=180, freq="D")
    base = 100 + np.cumsum(np.random.normal(0, 1.2, size=len(dates)))
    forecast = base + np.random.normal(0.3, 1.0, size=len(dates))
    df = pd.DataFrame({
        "target_date": dates,
        "actual_price": base,
        "forecast_price": forecast
    })
    return df

def _recommendation_to_signal(value):
    if value is None:
        return 0
    token = str(value).strip().upper()
    if token in {"BUY", "UP", "LONG", "상승", "매수"}:
        return 1
    if token in {"SELL", "DOWN", "SHORT", "하락", "매도"}:
        return -1
    return 0

def load_backtest_data(commodity, start_date, end_date):
    rec_df = get_test_commodity_data(
        commodity=commodity,
        start_date=start_date,
        end_date=end_date,
    )
    if rec_df.empty:
        return pd.DataFrame()

    price_df = get_performance_data(
        start_date=start_date,
        end_date=end_date,
        commodity=commodity,
    )
    if price_df.empty:
        return pd.DataFrame()

    rec_df = rec_df.copy()
    price_df = price_df.copy()
    rec_df["target_date"] = pd.to_datetime(rec_df["target_date"]).dt.date
    price_df["target_date"] = pd.to_datetime(price_df["target_date"]).dt.date
    price_df = price_df[["target_date", "actual_price"]].drop_duplicates(subset=["target_date"])

    merged = rec_df.merge(price_df, on="target_date", how="left").dropna(subset=["actual_price"])
    if merged.empty:
        return pd.DataFrame()

    merged["signal"] = merged["recommendation"].map(_recommendation_to_signal)

    # recommendation이 비어있으면 확률 최댓값으로 보정
    if {"p_buy", "p_hold", "p_sell"}.issubset(merged.columns):
        best_idx = merged[["p_buy", "p_hold", "p_sell"]].astype(float).idxmax(axis=1)
        fallback_signal = best_idx.map({"p_buy": 1, "p_hold": 0, "p_sell": -1}).fillna(0)
        merged["signal"] = np.where(merged["signal"] == 0, fallback_signal, merged["signal"])

    # 기존 백테스트 함수 호환용 가상 예측가격 컬럼
    merged["forecast_price"] = merged["actual_price"] * (1 + merged["signal"] * 0.01)
    merged["target_date"] = pd.to_datetime(merged["target_date"])
    return merged.sort_values("target_date").reset_index(drop=True)


st.set_page_config(page_title="Backtesting Strategy", layout="wide", initial_sidebar_state="collapsed")

# 고정 운용 파라미터 (UI 노출 제거)
DEFAULT_PROB_THRESHOLD = 0.50
DEFAULT_CONFIDENCE_GAMMA = 2.2

def apply_custom_style():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        
        :root {
            --bg-deep: #0a0e17; 
            --bg-card: rgba(17, 25, 40, 0.75); 
            --accent-primary: #00e5ff;
            --accent-success: #00c853;
            --accent-error: #ff1744;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --border-thin: rgba(255, 255, 255, 0.08);
        }

        [data-testid="stAppViewContainer"] {
            background: 
                radial-gradient(circle at 10% 10%, rgba(0, 229, 255, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 90%, rgba(0, 200, 83, 0.03) 0%, transparent 40%),
                var(--bg-deep);
            font-family: 'Inter', sans-serif;
            color: var(--text-main);
        }

        .custom-card {
            background: var(--bg-card);
            border: 1px solid var(--border-thin);
            border-radius: 6px;
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

        .score-highlight {
            color: var(--accent-primary);
            font-weight: 800;
        }

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
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin-top: 4px;
            margin-bottom: 10px;
        }
        .kpi-card {
            background: rgba(15, 23, 42, 0.58);
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 10px;
            padding: 12px 14px;
        }
        .kpi-title {
            font-size: 0.88rem;
            color: #cbd5e1;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .kpi-value {
            font-size: 1.7rem;
            font-weight: 800;
            line-height: 1.1;
            font-variant-numeric: tabular-nums;
        }
        .kpi-value-sm {
            font-size: 1.45rem;
            font-weight: 800;
            line-height: 1.1;
            font-variant-numeric: tabular-nums;
        }
        .kpi-pos { color: #22c55e; }
        .kpi-neg { color: #ef4444; }
        .kpi-neu { color: #e2e8f0; }
        .kpi-note {
            display: inline-flex;
            align-items: center;
            margin-top: 8px;
            border-radius: 999px;
            padding: 4px 8px;
            font-size: 0.76rem;
            font-weight: 700;
        }
        .kpi-note-pos {
            color: #22c55e;
            background: rgba(34, 197, 94, 0.14);
            border: 1px solid rgba(34, 197, 94, 0.34);
        }
        .kpi-note-neg {
            color: #ef4444;
            background: rgba(239, 68, 68, 0.14);
            border: 1px solid rgba(239, 68, 68, 0.34);
        }
        .kpi-note-neu {
            color: #cbd5e1;
            background: rgba(148, 163, 184, 0.16);
            border: 1px solid rgba(148, 163, 184, 0.34);
        }
        .mini-kpi {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(15, 23, 42, 0.4);
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 999px;
            padding: 6px 10px;
            margin-right: 8px;
            margin-bottom: 6px;
            font-size: 0.86rem;
            color: #cbd5e1;
        }
        .mini-kpi b {
            font-size: 0.92rem;
            font-weight: 800;
        }
        .summary-line {
            font-family: 'Inter', sans-serif;
            font-size: 0.96rem;
            font-weight: 600;
            color: #cbd5e1;
            letter-spacing: 0.01em;
            margin-top: 4px;
        }
        .summary-line .gap-pos { color: #22c55e; font-weight: 800; }
        .summary-line .gap-neg { color: #ef4444; font-weight: 800; }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%) !important;
            border-right: 1px solid var(--border-thin) !important;
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 {
            color: #f8fafc !important;
            font-weight: 700 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="radio"] label span {
            color: #f1f5f9 !important;
            font-weight: 700 !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: rgba(30, 41, 59, 0.9) !important;
            border: 1px solid rgba(148, 163, 184, 0.5) !important;
            color: #f8fafc !important;
        }
        [data-testid="stSidebar"] [data-baseweb="input"] input,
        [data-testid="stSidebar"] .stNumberInput input {
            color: #0f172a !important;
            font-weight: 700 !important;
            background: #f8fafc !important;
        }
        [data-testid="stSidebar"] [data-testid="stSlider"] label,
        [data-testid="stSidebar"] [data-testid="stSlider"] span,
        [data-testid="stSidebar"] [data-testid="stSlider"] p {
            color: #e2e8f0 !important;
            font-weight: 700 !important;
        }

        .stDataFrame, [data-testid="stDataFrame"] {
            background: rgba(15, 23, 42, 0.45) !important;
            border: 1px solid rgba(148, 163, 184, 0.3) !important;
        }
        [data-testid="stDataFrame"] th, .stDataFrame th {
            color: #f8fafc !important;
            background: rgba(30, 41, 59, 0.85) !important;
            font-weight: 800 !important;
        }
        [data-testid="stDataFrame"] td, .stDataFrame td {
            color: #e2e8f0 !important;
        }
        table th {
            color: #f8fafc !important;
            background: rgba(30, 41, 59, 0.85) !important;
        }
        table td {
            color: #e2e8f0 !important;
        }

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
        </style>
    """, unsafe_allow_html=True)

apply_custom_style()

def card_begin(title):
    st.markdown(f'<div class="custom-card"><div class="card-header">{title}</div>', unsafe_allow_html=True)

def card_end():
    st.markdown('</div>', unsafe_allow_html=True)

# 사이드바 입력
with st.sidebar:
    st.markdown("### 🌾 종목 설정")
    commodity_label = st.radio(
        "종목 선택",
        ["옥수수", "밀", "대두"],
        key="bt_commodity_label",
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
        key="bt_range_mode",
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
        start_date = st.date_input("시작일", today, key="bt_start_date")
        end_date = st.date_input("종료일", today, key="bt_end_date")

    st.markdown("### ⚙️ 운용 설정")
    seed = st.number_input("투자 원금 ($)", value=10000)
    fee = st.slider("거래 수수료 (%)", 0.0, 0.5, 0.1)
    st.caption(
        f"전략 파라미터 고정: 신호 기준 {DEFAULT_PROB_THRESHOLD:.2f} | 확신도 강도 {DEFAULT_CONFIDENCE_GAMMA:.1f}"
    )

# 데이터 로드 (BigQuery decision_meta + price join)
df = load_backtest_data(commodity=commodity, start_date=start_date, end_date=end_date)
if df.empty:
    st.warning("선택한 기간/종목에 백테스트 데이터가 없습니다. (decision_meta + 가격데이터 조인)")
    st.stop()

selected_threshold = DEFAULT_PROB_THRESHOLD
selected_gamma = DEFAULT_CONFIDENCE_GAMMA
results = calculate_backtest_comparison(
    df,
    initial_investment=seed,
    fee_pct=fee,
    prob_threshold=selected_threshold,
    confidence_gamma=selected_gamma,
)

if results is None:
    st.warning("기본 모델/LLM 최종 의사결정/가격 데이터가 충분하지 않아 백테스트를 계산할 수 없습니다.")
    st.stop()

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

st.markdown(
    f"""
    <div class="page-title">
        <span class="title-badge"></span>
        백테스팅 성과 리포트
        <span class="commodity-tag">
            <svg class="commodity-icon" viewBox="0 0 16 16">
                {commodity_svg}
            </svg>
            {commodity_label_upper}
        </span>
        <span class="title-sub">{start_date} ~ {end_date} · decision_meta</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "전략 비교: 기본 모델 의사결정, LLM 최종 의사결정, 기준전략"
)

# [KPI] 핵심 지표 3개 (간결 + 색상 강조)
alpha = float(results["alpha_pct"])
final_ret = float(results["returns_pct"]["v2"])
mdd = float(results["mdd_pct"])
hit_rate = float(results["hit_rate_pct"])

def _cls(v, positive_good=True):
    if v is None:
        return "kpi-neu"
    if positive_good:
        return "kpi-pos" if v >= 0 else "kpi-neg"
    return "kpi-pos" if v <= 0 else "kpi-neg"

def _note_cls(v, positive_good=True):
    if v is None:
        return "kpi-note-neu"
    if positive_good:
        return "kpi-note-pos" if v >= 0 else "kpi-note-neg"
    return "kpi-note-pos" if v <= 0 else "kpi-note-neg"

if alpha >= 0:
    alpha_note = "긍정: 기준전략 대비 개선"
else:
    alpha_note = "주의: 기준전략 대비 열위"

if final_ret >= 0:
    ret_note = "긍정: 기간 누적 수익"
else:
    ret_note = "주의: 기간 누적 손실"

if abs(mdd) <= 5.0:
    mdd_note = "긍정: 낙폭 관리 양호"
    mdd_note_cls = "kpi-note-pos"
else:
    mdd_note = "주의: 낙폭 확대 구간 존재"
    mdd_note_cls = "kpi-note-neg"

if hit_rate >= 55.0:
    hit_note = "긍정: 방향성 포착"
    hit_note_cls = "kpi-note-pos"
elif hit_rate >= 50.0:
    hit_note = "중립: 보합 수준"
    hit_note_cls = "kpi-note-neu"
else:
    hit_note = "주의: 정확도 보완 필요"
    hit_note_cls = "kpi-note-neg"

st.markdown(
    f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">전략 개선폭</div>
            <div class="kpi-value {_cls(alpha, True)}">{alpha:+.2f}%p</div>
            <span class="kpi-note {_note_cls(alpha, True)}">{alpha_note}</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">최종 전략 수익률</div>
            <div class="kpi-value {_cls(final_ret, True)}">{final_ret:+.2f}%</div>
            <span class="kpi-note {_note_cls(final_ret, True)}">{ret_note}</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">리스크(최대낙폭)</div>
            <div class="kpi-value {_cls(mdd, False)}">{mdd:.2f}%</div>
            <span class="kpi-note {mdd_note_cls}">{mdd_note}</span>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">방향 적중률</div>
            <div class="kpi-value-sm {_cls(hit_rate - 50.0, True)}">{hit_rate:.1f}%</div>
            <span class="kpi-note {hit_note_cls}">{hit_note}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# [Mini KPI] 방어 성격 요약
baseline_ret = float(results["returns_pct"]["baseline"])
final_ret = float(results["returns_pct"]["v2"])
baseline_mdd = float(results["drawdown_curves"]["baseline"].min() * 100.0)
final_mdd = float(results["mdd_pct"])

loss_reduction = None
if baseline_ret < 0 and final_ret < 0:
    loss_reduction = ((abs(baseline_ret) - abs(final_ret)) / abs(baseline_ret)) * 100.0 if abs(baseline_ret) > 0 else 0.0
elif baseline_ret < 0 <= final_ret:
    loss_reduction = 100.0

defense_score = None
if abs(baseline_mdd) > 1e-9:
    defense_score = (1.0 - (abs(final_mdd) / abs(baseline_mdd))) * 100.0

loss_text = "N/A" if loss_reduction is None else f"{loss_reduction:.1f}%"
defense_text = "N/A" if defense_score is None else f"{defense_score:.1f}%"
loss_cls = _cls(loss_reduction, True) if loss_reduction is not None else "kpi-neu"
defense_cls = _cls(defense_score, True) if defense_score is not None else "kpi-neu"
st.markdown(
    f"""
    <div class="mini-kpi">손실 절감률 <b class="{loss_cls}">{loss_text}</b></div>
    <div class="mini-kpi">하락장 방어점수 <b class="{defense_cls}">{defense_text}</b></div>
    """,
    unsafe_allow_html=True,
)

initial_capital = float(results.get("initial_investment", seed))
final_value = initial_capital * (1.0 + float(results["returns_pct"]["v2"]) / 100.0)
baseline_value = initial_capital * (1.0 + float(results["returns_pct"]["baseline"]) / 100.0)
value_gap = final_value - baseline_value
gap_text = f"+${value_gap:,.0f}" if value_gap >= 0 else f"-${abs(value_gap):,.0f}"
gap_cls = "gap-pos" if value_gap >= 0 else "gap-neg"
st.markdown(
    f"""
    <div class="summary-line">
        평가금액: 최종 전략 ${final_value:,.0f} | 기준전략 ${baseline_value:,.0f} | 차이 <span class="{gap_cls}">{gap_text}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"수익률 비교: 기본 모델 {results['returns_pct']['v1']}% | 최종 전략 {results['returns_pct']['v2']}% | 기준전략 {results['returns_pct']['baseline']}%"
)
st.caption(
    f"거래 횟수: 기본 모델 {results['trade_count']['v1']}회 | 최종 전략 {results['trade_count']['v2']}회"
)

st.markdown("---")

# [Charts] 자산 곡선, 낙폭, 매매 시점
card_begin("📈 전략 비교 (기본 모델 vs LLM 최종)")
fig_equity, fig_dd, fig_signal = draw_backtest_charts(df, results)
st.caption("실선: 누적 수익률(%) / 녹색 점선: LLM 포지션 비중(%) / 삼각형: 매수·매도 시점")
st.plotly_chart(fig_equity, use_container_width=True)
card_end()
