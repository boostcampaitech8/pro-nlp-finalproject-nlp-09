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


def _safe_sharpe(daily_returns: pd.Series) -> float:
    std = daily_returns.std()
    if std is None or std == 0 or np.isnan(std):
        return 0.0
    return float((daily_returns.mean() / std) * np.sqrt(252))


def _simulate_probability_strategy(
    df: pd.DataFrame,
    stage: str,
    initial_investment: float,
    fee_rate: float,
    prob_threshold: float,
    confidence_gamma: float = 2.0,
):
    stage_df = df[df["stage"].astype(str).str.lower() == stage.lower()].copy()
    if stage_df.empty:
        return None

    stage_df["target_date"] = pd.to_datetime(stage_df["target_date"])
    stage_df = stage_df.sort_values(["target_date", "ingested_at"], ascending=[True, False])
    stage_df = stage_df.drop_duplicates(subset=["target_date"], keep="first")
    stage_df = stage_df.sort_values("target_date")

    stage_df["p_buy"] = pd.to_numeric(stage_df["p_buy"], errors="coerce").fillna(0.0)
    stage_df["p_sell"] = pd.to_numeric(stage_df["p_sell"], errors="coerce").fillna(0.0)

    # BUY/SELL이면 항상 거래가 발생하도록 신호 완화 (p_buy vs p_sell 비교)
    net_signal = (stage_df["p_buy"] - stage_df["p_sell"]).clip(lower=-1.0, upper=1.0)
    stage_df["signal"] = np.where(net_signal > 0, 1, np.where(net_signal < 0, -1, 0))

    # 확신도 기반 거래 강도(0~1)
    # - 낮은 확신: 소량 거래
    # - 높은 확신: 대량 거래
    min_trade_intensity = 0.06
    confidence = net_signal.abs().clip(lower=0.0, upper=1.0)
    stage_df["trade_intensity"] = (
        min_trade_intensity + (1.0 - min_trade_intensity) * (confidence ** confidence_gamma)
    ).clip(lower=min_trade_intensity, upper=1.0)

    prices = pd.to_numeric(stage_df["actual_price"], errors="coerce")
    stage_df = stage_df.loc[prices.notna()].copy()
    stage_df["actual_price"] = prices.loc[prices.notna()].astype(float)
    if stage_df.empty:
        return None

    cash = float(initial_investment)
    units = 0.0
    equities = []
    weights = []
    buy_dates = []
    sell_dates = []

    for _, row in stage_df.iterrows():
        price = float(row["actual_price"])
        signal = int(row["signal"])
        trade_intensity = float(row["trade_intensity"])
        dt = row["target_date"]

        # 신호 기반 비중 조절 리밸런싱
        equity_before = cash + units * price
        if equity_before <= 0:
            equities.append(0.0)
            weights.append(0.0)
            continue

        current_position_value = units * price
        current_weight = current_position_value / equity_before

        if signal == 1:
            target_weight = min(1.0, current_weight + trade_intensity * (1.0 - current_weight))
        elif signal == -1:
            target_weight = max(0.0, current_weight - trade_intensity * current_weight)
        else:
            target_weight = current_weight

        target_position_value = equity_before * target_weight
        delta_value = target_position_value - current_position_value

        if delta_value > 1e-9:
            # 매수: 수수료를 포함한 현금 소요를 반영
            required_cash = delta_value / max(1.0 - fee_rate, 1e-9)
            spend_cash = min(cash, required_cash)
            bought_value = spend_cash * (1.0 - fee_rate)
            units += bought_value / price
            cash -= spend_cash
            if signal == 1:
                buy_dates.append(dt)
        elif delta_value < -1e-9:
            # 매도: 목표 비중까지 일부 또는 전량 축소
            sell_value = min(current_position_value, abs(delta_value))
            units -= sell_value / price
            cash += sell_value * (1.0 - fee_rate)
            if signal == -1:
                sell_dates.append(dt)

        equity_after = cash + units * price
        equities.append(equity_after)
        weights.append((units * price) / equity_after if equity_after > 0 else 0.0)

    equity_series = pd.Series(equities, index=stage_df["target_date"], name=f"{stage}_equity")
    daily_returns = equity_series.pct_change().fillna(0.0)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max

    final_return = float((equity_series.iloc[-1] / initial_investment) - 1.0)
    sharpe = _safe_sharpe(daily_returns)
    mdd = float(drawdown.min()) if len(drawdown) else 0.0

    next_price = stage_df["actual_price"].shift(-1)
    next_ret = (next_price - stage_df["actual_price"]) / stage_df["actual_price"]
    non_hold = stage_df["signal"] != 0
    hit_mask = non_hold & next_ret.notna()
    if hit_mask.any():
        hit_rate = float(((stage_df.loc[hit_mask, "signal"] * next_ret.loc[hit_mask]) > 0).mean() * 100.0)
    else:
        hit_rate = 0.0

    return {
        "stage": stage,
        "stage_df": stage_df,
        "equity_series": equity_series,
        "weight_series": pd.Series(weights, index=stage_df["target_date"], name=f"{stage}_weight"),
        "drawdown_series": drawdown,
        "daily_returns": daily_returns,
        "final_return": final_return,
        "sharpe": sharpe,
        "mdd": mdd,
        "hit_rate": hit_rate,
        "buy_dates": buy_dates,
        "sell_dates": sell_dates,
        "trade_count": len(buy_dates) + len(sell_dates),
    }


def _simulate_buy_and_hold(price_df: pd.DataFrame, initial_investment: float, fee_rate: float):
    if price_df.empty:
        return None
    base_df = price_df.copy().sort_values("target_date")
    base_df["target_date"] = pd.to_datetime(base_df["target_date"])
    base_df = base_df.drop_duplicates(subset=["target_date"], keep="first")
    base_df["actual_price"] = pd.to_numeric(base_df["actual_price"], errors="coerce")
    base_df = base_df.dropna(subset=["actual_price"])
    if base_df.empty:
        return None

    first_price = float(base_df["actual_price"].iloc[0])
    units = (initial_investment * (1 - fee_rate)) / first_price
    equity_series = pd.Series(
        units * base_df["actual_price"].astype(float).values,
        index=base_df["target_date"],
        name="buy_hold_equity",
    )
    daily_returns = equity_series.pct_change().fillna(0.0)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max

    return {
        "equity_series": equity_series,
        "drawdown_series": drawdown,
        "daily_returns": daily_returns,
        "final_return": float((equity_series.iloc[-1] / initial_investment) - 1.0),
        "sharpe": _safe_sharpe(daily_returns),
        "mdd": float(drawdown.min()) if len(drawdown) else 0.0,
    }


def calculate_backtest_comparison(
    df: pd.DataFrame,
    initial_investment: float,
    fee_pct: float = 0.1,
    prob_threshold: float = 0.6,
    confidence_gamma: float = 2.0,
):
    """
    v1/v2/Buy&Hold 비교 백테스팅.
    - 진입: p_buy >= threshold
    - 청산: p_sell >= threshold
    """
    if df is None or df.empty:
        return None

    fee_rate = float(fee_pct) / 100.0
    work_df = df.copy()
    work_df["target_date"] = pd.to_datetime(work_df["target_date"])

    v1 = _simulate_probability_strategy(
        work_df, "v1", initial_investment, fee_rate, prob_threshold, confidence_gamma
    )
    v2 = _simulate_probability_strategy(
        work_df, "v2", initial_investment, fee_rate, prob_threshold, confidence_gamma
    )

    # Baseline은 v2 가격을 우선 사용, 없으면 전체 날짜가격 사용
    if v2 is not None:
        base_price_df = v2["stage_df"][["target_date", "actual_price"]]
    elif v1 is not None:
        base_price_df = v1["stage_df"][["target_date", "actual_price"]]
    else:
        base_price_df = work_df[["target_date", "actual_price"]]
    baseline = _simulate_buy_and_hold(base_price_df, initial_investment, fee_rate)

    if v1 is None or v2 is None or baseline is None:
        return None

    alpha_pct = (v2["final_return"] - baseline["final_return"]) * 100.0
    v2_improvement = v2["sharpe"] - v1["sharpe"]

    return {
        "alpha_pct": round(alpha_pct, 2),
        "v2_improvement": round(v2_improvement, 3),
        "hit_rate_pct": round(v2["hit_rate"], 1),
        "mdd_pct": round(v2["mdd"] * 100.0, 2),
        "initial_investment": float(initial_investment),
        "returns_pct": {
            "v1": round(v1["final_return"] * 100.0, 2),
            "v2": round(v2["final_return"] * 100.0, 2),
            "baseline": round(baseline["final_return"] * 100.0, 2),
        },
        "sharpe": {
            "v1": round(v1["sharpe"], 3),
            "v2": round(v2["sharpe"], 3),
            "baseline": round(baseline["sharpe"], 3),
        },
        "strategy_curves": {
            "v1": v1["equity_series"],
            "v2": v2["equity_series"],
            "baseline": baseline["equity_series"],
        },
        "weight_curves": {
            "v1": v1["weight_series"],
            "v2": v2["weight_series"],
        },
        "drawdown_curves": {
            "v1": v1["drawdown_series"],
            "v2": v2["drawdown_series"],
            "baseline": baseline["drawdown_series"],
        },
        "signals": {
            "v1": {"buy_dates": v1["buy_dates"], "sell_dates": v1["sell_dates"]},
            "v2": {"buy_dates": v2["buy_dates"], "sell_dates": v2["sell_dates"]},
        },
        "trade_count": {
            "v1": v1["trade_count"],
            "v2": v2["trade_count"],
        },
        "price_df": base_price_df.sort_values("target_date").reset_index(drop=True),
    }
