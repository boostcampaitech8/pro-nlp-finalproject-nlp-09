import numpy as np
import pandas as pd

def calculate_metrics(df):
    """
    성능 평가를 위한 핵심 지표(MAE, Hit Rate 등)를 계산합니다.
    """
    # 1. MAE (평균 절대 오차) 계산
    mae = abs(df['actual_price'] - df['forecast_price']).mean()
    mse = ((df['actual_price'] - df['forecast_price']) ** 2).mean()
    rmse = mse ** 0.5
    # 2. 방향 적중률 (Hit Rate) 계산
    # (오늘 시가 - 어제 시가)의 방향과 (예측가 - 어제 시가)의 방향이 같은지 확인
    hit_count = 0
    for i in range(1, len(df)):
        # 실제 방향
        actual_diff = df['actual_price'].iloc[i] - df['actual_price'].iloc[i-1]
        # 예측 방향
        forecast_diff = df['forecast_price'].iloc[i] - df['actual_price'].iloc[i-1]
        
        if (actual_diff * forecast_diff) > 0: # 둘의 부호가 같으면 적중
            hit_count += 1
            
    hit_rate = (hit_count / (len(df) - 1)) * 100 if len(df) > 1 else 0
    
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "hit_rate": round(hit_rate, 1)
    }

def calculate_kpi_metrics(df):
    # Backward-compatible alias used by app.py
    return calculate_metrics(df)


def calculate_backtest_metrics(df, initial_investment):
    # 1. 자산 변화 계산 (수익률 기반)
    # 단순화: 예측 방향이 맞으면 그날 변동폭만큼 수익, 틀리면 손실 가정
    df['daily_return'] = (df['actual_price'].pct_change().shift(-1)).fillna(0)
    # 모델 시그널 (상승 예측 시 1, 하락 예측 시 -1)
    df['signal'] = np.where(df['forecast_price'] > df['actual_price'], 1, -1)
    df['strategy_return'] = df['signal'] * df['daily_return']
    
    # 누적 수익 및 자산 가치
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    df['equity_curve'] = df['cumulative_return'] * initial_investment
    
    # 2. 핵심 지표 계산
    total_return = (df['equity_curve'].iloc[-1] / initial_investment) - 1
    annualized_return = (1 + total_return) ** (365 / len(df)) - 1 # CAGR
    
    # 변동성 및 Sharpe Ratio (무위험 수익률 0 가정)
    volatility = df['strategy_return'].std() * np.sqrt(252)
    sharpe_ratio = (df['strategy_return'].mean() / df['strategy_return'].std()) * np.sqrt(252) if df['strategy_return'].std() != 0 else 0
    
    # MDD (최대 낙폭)
    rolling_max = df['equity_curve'].cummax()
    drawdown = (df['equity_curve'] - rolling_max) / rolling_max
    mdd = drawdown.min()
    
    # Profit Factor (총 이익 / 총 손실)
    gains = df[df['strategy_return'] > 0]['strategy_return'].sum()
    losses = abs(df[df['strategy_return'] < 0]['strategy_return'].sum())
    profit_factor = gains / losses if losses != 0 else np.inf
    
    # 3. Model Quality Score 
    # 가공: Sharpe(40%) + Stability(MDD 기반, 30%) + WinRate(30%)
    win_rate = (df['strategy_return'] > 0).sum() / len(df)
    stability = 1 - abs(mdd) # MDD가 적을수록 1에 가까움
    quality_score = (sharpe_ratio * 0.4) + (stability * 0.3) + (win_rate * 0.3)
    
    return {
        "cagr": round(annualized_return * 100, 2),
        "sharpe": round(sharpe_ratio, 2),
        "mdd": round(mdd * 100, 2),
        "profit_factor": round(profit_factor, 2),
        "win_rate": round(win_rate * 100, 1),
        "final_value": int(df['equity_curve'].iloc[-1]),
        "score": round(min(max(quality_score * 20, 0), 100), 1), # 100점 만점 환산
        "drawdown_series": drawdown,
        "equity_series": df['equity_curve']
    }