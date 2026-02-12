import plotly.graph_objects as go
from plotly.subplots import make_subplots

def draw_main_chart(df):
    fig = go.Figure()
    grid_color = "rgba(148, 163, 184, 0.18)"
    axis_color = "#ffffff"
    # 신뢰구간 반투명 밴드
    fig.add_trace(go.Scatter(
        x=df['target_date'].tolist() + df['target_date'].tolist()[::-1],
        y=df['yhat_upper'].tolist() + df['yhat_lower'].tolist()[::-1],
        fill='toself', fillcolor='rgba(56, 189, 248, 0.12)',
        line=dict(color='rgba(255,255,255,0)'), name="Confidence Interval",
        hoverinfo="skip"
    ))
    # 실제가
    fig.add_trace(go.Scatter(x=df['target_date'], y=df['actual_price'], 
                             mode='lines+markers', name="Actual Price",
                             line=dict(color='#38bdf8', width=3),
                             marker=dict(size=8, color="#38bdf8", line=dict(color="rgba(15, 23, 42, 0.9)", width=1))))
    # 예측가
    fig.add_trace(go.Scatter(x=df['target_date'], y=df['forecast_price'], 
                             mode='lines+markers', name="LLM Forecast",
                             line=dict(color='#f59e0b', width=2, dash='dot'),
                             marker=dict(color='#f59e0b', size=10, symbol="diamond", line=dict(color="rgba(15, 23, 42, 0.9)", width=1))))
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=520,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        font=dict(color=axis_color),
        xaxis=dict(tickfont=dict(color=axis_color), title=dict(font=dict(color=axis_color))),
        yaxis=dict(tickfont=dict(color=axis_color), title=dict(font=dict(color=axis_color))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=axis_color))
    )
    fig.update_traces(hoverlabel=dict(bgcolor="rgba(15, 23, 42, 0.9)", font_color="#e2e8f0"))
    fig.update_xaxes(showgrid=True, gridcolor=grid_color, zeroline=False, showline=False)
    fig.update_yaxes(showgrid=True, gridcolor=grid_color, zeroline=False, showline=False)
    return fig

def draw_sentiment_chart(df):
    required_cols = {"target_date", "sentiment"}
    if not required_cols.issubset(df.columns):
        return None

    plot_df = df.copy()
    plot_df["target_date"] = plot_df["target_date"]
    plot_df["sentiment"] = plot_df["sentiment"]
    plot_df = plot_df.dropna(subset=["target_date", "sentiment"]).sort_values("target_date")
    if plot_df.empty:
        return None

    # 같은 날짜 중복 포인트 제거
    plot_df = plot_df.groupby("target_date", as_index=False)["sentiment"].mean()
    plot_df["sentiment_ma7"] = plot_df["sentiment"].rolling(7, min_periods=1).mean()

    grid_color = "rgba(148, 163, 184, 0.18)"
    axis_color = "#ffffff"
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["target_date"],
            y=plot_df["sentiment"],
            name="일별 감성 점수",
            marker_color=["#22c55e" if x >= 0 else "#ef4444" for x in plot_df["sentiment"]],
            marker_line=dict(color="rgba(15, 23, 42, 0.9)", width=1),
            opacity=0.65,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["target_date"],
            y=plot_df["sentiment_ma7"],
            mode="lines",
            name="7일 이동평균",
            line=dict(color="#38bdf8", width=2.6),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(color=axis_color),
        legend=dict(font=dict(color=axis_color)),
        xaxis=dict(tickfont=dict(color=axis_color), title=dict(font=dict(color=axis_color))),
        yaxis=dict(tickfont=dict(color=axis_color), title=dict(font=dict(color=axis_color)))
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False)
    fig.update_yaxes(
        title_text="감성 점수",
        showgrid=True,
        gridcolor=grid_color,
        zeroline=True,
        zerolinecolor="rgba(148, 163, 184, 0.5)",
    )
    return fig


def draw_sentiment_lead_lag_chart(df):
    required_cols = {"target_date", "sentiment", "actual_price"}
    if not required_cols.issubset(df.columns):
        return None

    plot_df = df.copy()
    plot_df["target_date"] = plot_df["target_date"]
    plot_df["sentiment"] = plot_df["sentiment"]
    plot_df["actual_price"] = plot_df["actual_price"]
    plot_df = plot_df.dropna(subset=["target_date", "sentiment", "actual_price"]).sort_values("target_date")
    plot_df = (
        plot_df.groupby("target_date", as_index=False)
        .agg(sentiment=("sentiment", "mean"), actual_price=("actual_price", "mean"))
    )
    if plot_df.empty:
        return None

    grid_color = "rgba(148, 163, 184, 0.18)"
    axis_color = "#ffffff"
    bar_colors = ["#22c55e" if x >= 0 else "#ef4444" for x in plot_df["sentiment"]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=plot_df["target_date"],
            y=plot_df["sentiment"],
            name="일별 감성 점수",
            marker_color=bar_colors,
            opacity=0.75,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["target_date"],
            y=plot_df["actual_price"],
            mode="lines+markers",
            name="실제 종가",
            line=dict(color="#60a5fa", width=2.8),
            marker=dict(size=6),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        font=dict(color=axis_color),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor=grid_color, zeroline=False, showline=False)
    fig.update_yaxes(
        title_text="감성 점수 (pos_ratio - neg_ratio)",
        showgrid=True,
        gridcolor=grid_color,
        zeroline=True,
        zerolinecolor="rgba(148, 163, 184, 0.5)",
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="실제 종가",
        showgrid=False,
        zeroline=False,
        secondary_y=True,
    )
    return fig


def draw_volume_volatility_bubble(df):
    required_cols = {"target_date", "sentiment", "actual_price", "news_count"}
    if not required_cols.issubset(df.columns):
        return None

    plot_df = df.copy()
    plot_df["target_date"] = plot_df["target_date"]
    plot_df["actual_price"] = plot_df["actual_price"]
    plot_df["sentiment"] = plot_df["sentiment"]
    plot_df["news_count"] = plot_df["news_count"]
    plot_df = plot_df.dropna(subset=["target_date", "actual_price", "sentiment", "news_count"]).sort_values("target_date")
    plot_df = (
        plot_df.groupby("target_date", as_index=False)
        .agg(
            actual_price=("actual_price", "mean"),
            sentiment=("sentiment", "mean"),
            news_count=("news_count", "sum"),
        )
    )
    if len(plot_df) < 2:
        return None

    plot_df["price_change_pct"] = plot_df["actual_price"].pct_change() * 100.0
    plot_df = plot_df.dropna(subset=["price_change_pct"])
    if plot_df.empty:
        return None

    max_news = max(float(plot_df["news_count"].max()), 1.0)
    sizes = (plot_df["news_count"] / max_news) * 42 + 8

    fig = go.Figure(
        data=[
            go.Scatter(
                x=plot_df["price_change_pct"],
                y=plot_df["sentiment"],
                mode="markers",
                name="일별 포인트",
                marker=dict(
                    size=sizes,
                    color=plot_df["sentiment"],
                    colorscale="RdYlGn",
                    reversescale=False,
                    showscale=True,
                    colorbar=dict(title="감성 점수"),
                    opacity=0.78,
                    line=dict(color="rgba(15, 23, 42, 0.8)", width=1),
                ),
                customdata=plot_df[["target_date", "news_count"]],
                hovertemplate=(
                    "날짜: %{customdata[0]}<br>"
                    "가격 변동률: %{x:.2f}%<br>"
                    "감성 점수: %{y:.2f}<br>"
                    "뉴스 수: %{customdata[1]}건<extra></extra>"
                ),
            )
        ]
    )

    grid_color = "rgba(148, 163, 184, 0.18)"
    axis_color = "#ffffff"
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color=axis_color),
        xaxis=dict(title="일별 가격 변동률 (%)"),
        yaxis=dict(title="일별 감성 점수"),
    )
    fig.update_xaxes(showgrid=True, gridcolor=grid_color, zeroline=True, zerolinecolor="rgba(148, 163, 184, 0.5)")
    fig.update_yaxes(showgrid=True, gridcolor=grid_color, zeroline=True, zerolinecolor="rgba(148, 163, 184, 0.5)")
    return fig

def draw_contribution_chart(df):
    if "target_date" not in df.columns:
        return None

    candidates = [
        ("trend", "장기 방향"),
        ("yearly", "연간 패턴"),
        ("weekly", "주간 패턴"),
        ("extra_regressors_multiplicative", "외부 영향"),
        ("Volume_lag5_effect", "거래량 영향(5일)"),
        ("EMA_lag2_effect", "EMA 영향(2일)"),
    ]
    available = [(col, label) for col, label in candidates if col in df.columns]
    if not available:
        return None

    # 상위 5개 기여도만 사용 (정규화 후 토글로 표시)
    contrib_scores = {col: df[col].abs().mean() for col, _ in available}
    top_cols = sorted(contrib_scores, key=contrib_scores.get, reverse=True)[:5]
    label_map = {col: label for col, label in available}

    fig = go.Figure()
    palette = ["#ef4444", "#f97316", "#facc15", "#22c55e", "#3b82f6"]
    for idx, col in enumerate(top_cols):
        base = df[col].iloc[0] if len(df[col]) else 1.0
        base = base if base != 0 else 1.0
        series_index = (df[col] / base) * 100.0
        label = label_map.get(col, col)
        fig.add_trace(
            go.Scatter(
                x=df["target_date"],
                y=series_index,
                mode="lines",
                name=label,
                line=dict(width=2.0, color=palette[idx % len(palette)]),
                visible="legendonly",
            )
        )

    if "actual_price" in df.columns:
        base_actual = df["actual_price"].iloc[0] if len(df["actual_price"]) else 1.0
        base_actual = base_actual if base_actual != 0 else 1.0
        actual_index = (df["actual_price"] / base_actual) * 100.0
        fig.add_trace(
            go.Scatter(
                x=df["target_date"],
                y=actual_index,
                mode="lines",
                name="실제 가격 흐름",
                line=dict(color="#e2e8f0", width=3.0),
            )
        )

    grid_color = "rgba(148, 163, 184, 0.18)"
    axis_color = "#ffffff"
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        font=dict(color=axis_color),
        xaxis=dict(tickfont=dict(color=axis_color), title=dict(font=dict(color=axis_color))),
        yaxis=dict(
            tickfont=dict(color=axis_color),
            title=dict(text="흐름 지수(첫날=100)", font=dict(color=axis_color)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=axis_color)),
    )
    fig.update_traces(hoverlabel=dict(bgcolor="rgba(15, 23, 42, 0.9)", font_color="#e2e8f0"))
    fig.update_xaxes(showgrid=True, gridcolor=grid_color, zeroline=False, showline=False)
    fig.update_yaxes(showgrid=True, gridcolor=grid_color, zeroline=False, showline=False)
    return fig

def draw_performance_chart(df):
    # Backward-compatible alias used by app.py
    return draw_main_chart(df)

def draw_backtest_charts(df, results):
    grid_color = "rgba(148, 163, 184, 0.18)"
    axis_color = "#ffffff"

    # 신규 비교 백테스트 결과 포맷(기본 모델/LLM 최종/baseline)
    strategy_curves = results.get("strategy_curves")
    drawdown_curves = results.get("drawdown_curves")
    price_df = results.get("price_df")
    signals = results.get("signals", {})

    if strategy_curves and drawdown_curves and price_df is not None:
        initial_investment = float(results.get("initial_investment", 10000.0))
        weight_curves = results.get("weight_curves", {})
        fig_equity = make_subplots(specs=[[{"secondary_y": True}]])

        v1_return = (strategy_curves["v1"] / initial_investment - 1.0) * 100.0
        v2_return = (strategy_curves["v2"] / initial_investment - 1.0) * 100.0
        base_return = (strategy_curves["baseline"] / initial_investment - 1.0) * 100.0

        fig_equity.add_trace(
            go.Scatter(
                x=v1_return.index,
                y=v1_return.values,
                mode="lines",
                name="기본 모델 의사결정",
                line=dict(color="#60a5fa", width=2.5),
            ),
            secondary_y=False,
        )
        fig_equity.add_trace(
            go.Scatter(
                x=v2_return.index,
                y=v2_return.values,
                mode="lines",
                name="LLM 최종 의사결정",
                line=dict(color="#22d3ee", width=3.2),
            ),
            secondary_y=False,
        )
        fig_equity.add_trace(
            go.Scatter(
                x=base_return.index,
                y=base_return.values,
                mode="lines",
                name="Buy & Hold",
                line=dict(color="#f59e0b", width=2.0, dash="dot"),
            ),
            secondary_y=False,
        )

        v2_weight = weight_curves.get("v2")
        if v2_weight is not None and len(v2_weight):
            fig_equity.add_trace(
                go.Scatter(
                    x=v2_weight.index,
                    y=v2_weight.values * 100.0,
                    mode="lines",
                    name="LLM 포지션 비중(%)",
                    line=dict(color="#34d399", width=1.7, dash="dash"),
                    opacity=0.7,
                    visible="legendonly",
                ),
                secondary_y=True,
            )

        v2_buy_dates = set(signals.get("v2", {}).get("buy_dates", []))
        v2_sell_dates = set(signals.get("v2", {}).get("sell_dates", []))
        buy_mask = v2_return.index.isin(v2_buy_dates)
        sell_mask = v2_return.index.isin(v2_sell_dates)
        if buy_mask.any():
            fig_equity.add_trace(
                go.Scatter(
                    x=v2_return.index[buy_mask],
                    y=v2_return.values[buy_mask],
                    mode="markers",
                    name="LLM BUY 시점",
                    marker=dict(symbol="triangle-up", size=9, color="#22c55e"),
                    visible="legendonly",
                ),
                secondary_y=False,
            )
        if sell_mask.any():
            fig_equity.add_trace(
                go.Scatter(
                    x=v2_return.index[sell_mask],
                    y=v2_return.values[sell_mask],
                    mode="markers",
                    name="LLM SELL 시점",
                    marker=dict(symbol="triangle-down", size=9, color="#ef4444"),
                    visible="legendonly",
                ),
                secondary_y=False,
            )

        fig_dd = go.Figure()
        fig_dd.add_trace(
            go.Scatter(
                x=drawdown_curves["v1"].index,
                y=drawdown_curves["v1"].values * 100,
                mode="lines",
                name="기본 모델 낙폭",
                line=dict(color="#93c5fd", width=1.8),
            )
        )
        fig_dd.add_trace(
            go.Scatter(
                x=drawdown_curves["v2"].index,
                y=drawdown_curves["v2"].values * 100,
                mode="lines",
                name="LLM 최종 낙폭",
                line=dict(color="#38bdf8", width=2.4),
            )
        )

        price_plot = price_df.copy()
        price_plot["target_date"] = price_plot["target_date"]
        fig_signal = go.Figure()
        fig_signal.add_trace(
            go.Scatter(
                x=price_plot["target_date"],
                y=price_plot["actual_price"],
                mode="lines",
                name="Actual Price",
                line=dict(color="#e2e8f0", width=2.5),
            )
        )

        v2_buy_dates = set(signals.get("v2", {}).get("buy_dates", []))
        v2_sell_dates = set(signals.get("v2", {}).get("sell_dates", []))
        buy_mask = price_plot["target_date"].isin(v2_buy_dates)
        sell_mask = price_plot["target_date"].isin(v2_sell_dates)

        if buy_mask.any():
            fig_signal.add_trace(
                go.Scatter(
                    x=price_plot.loc[buy_mask, "target_date"],
                    y=price_plot.loc[buy_mask, "actual_price"],
                    mode="markers",
                    name="LLM 최종 BUY",
                    marker=dict(symbol="triangle-up", size=12, color="#22c55e"),
                )
            )
        if sell_mask.any():
            fig_signal.add_trace(
                go.Scatter(
                    x=price_plot.loc[sell_mask, "target_date"],
                    y=price_plot.loc[sell_mask, "actual_price"],
                    mode="markers",
                    name="LLM 최종 SELL",
                    marker=dict(symbol="triangle-down", size=12, color="#ef4444"),
                )
            )

        for fig, top_margin in ((fig_equity, 40), (fig_dd, 30), (fig_signal, 30)):
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=top_margin, b=20),
                hovermode="x unified",
                font=dict(color=axis_color),
                legend=dict(
                    font=dict(color="#e2e8f0", size=14),
                    bgcolor="rgba(15, 23, 42, 0.45)",
                ),
                xaxis=dict(tickfont=dict(color=axis_color), title=dict(font=dict(color=axis_color))),
                yaxis=dict(tickfont=dict(color=axis_color), title=dict(font=dict(color=axis_color))),
            )
            fig.update_traces(hoverlabel=dict(bgcolor="rgba(15, 23, 42, 0.9)", font_color="#e2e8f0"))
            fig.update_xaxes(showgrid=True, gridcolor=grid_color, zeroline=False, showline=False)
            fig.update_yaxes(showgrid=True, gridcolor=grid_color, zeroline=False, showline=False)
        fig_equity.update_yaxes(title_text="누적 수익률 (%)", secondary_y=False)
        fig_equity.update_yaxes(title_text="포지션 비중 (%)", range=[0, 100], secondary_y=True, showgrid=False)
        fig_dd.update_yaxes(title_text="Drawdown (%)")

        return fig_equity, fig_dd, fig_signal

    # 구버전 포맷 fallback
    if "target_date" in df.columns:
        x = df["target_date"]
    elif "date" in df.columns:
        x = df["date"]
    else:
        x = list(range(len(df)))

    equity = results.get("equity_series", df.get("equity_curve"))
    drawdown = results.get("drawdown_series")

    fig_equity = go.Figure()
    fig_equity.add_trace(
        go.Scatter(x=x, y=equity, mode="lines", name="Equity Curve", line=dict(color="#38bdf8", width=3))
    )
    fig_equity.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        font=dict(color=axis_color),
        xaxis=dict(tickfont=dict(color=axis_color), title=dict(font=dict(color=axis_color))),
        yaxis=dict(tickfont=dict(color=axis_color), title=dict(font=dict(color=axis_color))),
    )
    fig_equity.update_traces(hoverlabel=dict(bgcolor="rgba(15, 23, 42, 0.9)", font_color="#e2e8f0"))
    fig_equity.update_xaxes(showgrid=True, gridcolor=grid_color, zeroline=False, showline=False)
    fig_equity.update_yaxes(showgrid=True, gridcolor=grid_color, zeroline=False, showline=False)

    fig_dd = go.Figure()
    if drawdown is not None:
        fig_dd.add_trace(
            go.Scatter(x=x, y=drawdown, mode="lines", name="Drawdown", line=dict(color="#f59e0b", width=2))
        )
    fig_dd.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
        hovermode="x unified",
        font=dict(color=axis_color),
        xaxis=dict(tickfont=dict(color=axis_color), title=dict(font=dict(color=axis_color))),
        yaxis=dict(tickfont=dict(color=axis_color), title=dict(font=dict(color=axis_color))),
    )
    fig_dd.update_traces(hoverlabel=dict(bgcolor="rgba(15, 23, 42, 0.9)", font_color="#e2e8f0"))
    fig_dd.update_xaxes(showgrid=True, gridcolor=grid_color, zeroline=False, showline=False)
    fig_dd.update_yaxes(showgrid=True, gridcolor=grid_color, zeroline=True, zerolinecolor="rgba(148, 163, 184, 0.5)")

    return fig_equity, fig_dd, go.Figure()
