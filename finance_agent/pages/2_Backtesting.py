import streamlit as st
from src.analytics import calculate_backtest_metrics
from src.visualizer import draw_backtest_charts
import pandas as pd
import numpy as np

def load_mock_data():
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    base = 100 + np.cumsum(np.random.normal(0, 1.2, size=len(dates)))
    forecast = base + np.random.normal(0.3, 1.0, size=len(dates))
    df = pd.DataFrame({
        "target_date": dates,
        "actual_price": base,
        "forecast_price": forecast
    })
    return df

st.set_page_config(page_title="Backtesting Strategy", layout="wide", initial_sidebar_state="collapsed")

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

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%) !important;
            border-right: 1px solid var(--border-thin) !important;
        }
        [data-testid="stSidebar"] * {
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
    st.header("⚙️ Simulation Params")
    seed = st.number_input("투자 원금 ($)", value=10000)
    fee = st.slider("거래 수수료 (%)", 0.0, 0.5, 0.1)

# 데이터 로드 (Mock)
df = load_mock_data() # 기존 데이터 함수 사용
results = calculate_backtest_metrics(df, seed)

st.markdown(
    """
    <div class="page-title">
        <span class="title-badge"></span>
        백테스팅 성과 리포트
        <span class="title-sub">Mock 데이터 기반</span>
    </div>
    """,
    unsafe_allow_html=True,
)

card_begin("🎖️ Model Quality Score")
st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:10px; font-size:1.1rem;">
        <span class="score-highlight">{results['score']} / 100</span>
        <span style="color:#94a3b8; font-size:0.95rem;">수익성, 안정성, 리스크 대응력을 종합한 모델 등급입니다.</span>
    </div>
    """,
    unsafe_allow_html=True,
)
card_end()

# [KPI] 핵심 지표 4개
c1, c2, c3, c4 = st.columns(4)
c1.metric("CAGR (연수익률)", f"{results['cagr']}%")
c2.metric("Sharpe Ratio", results['sharpe'])
c3.metric("Max Drawdown", f"{results['mdd']}%", delta_color="inverse")
c4.metric("Profit Factor", results['profit_factor'])

st.markdown("---")

# [Charts] 자산 곡선 및 낙폭
card_begin("📈 Strategy Performance Analysis")
fig_equity, fig_dd = draw_backtest_charts(df, results)
st.plotly_chart(fig_equity, use_container_width=True)
st.plotly_chart(fig_dd, use_container_width=True)
card_end()
