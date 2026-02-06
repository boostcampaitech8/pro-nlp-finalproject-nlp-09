import plotly.graph_objects as go

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
    grid_color = "rgba(148, 163, 184, 0.18)"
    axis_color = "#ffffff"
    # 뉴스 감성 변화 막대 그래프
    fig = go.Figure(go.Bar(
        x=df['target_date'], y=df['sentiment'],
        marker_color=['#f59e0b' if x < 0 else '#38bdf8' for x in df['sentiment']],
        marker_line=dict(color="rgba(15, 23, 42, 0.9)", width=1)
    ))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(color=axis_color),
        xaxis=dict(tickfont=dict(color=axis_color), title=dict(font=dict(color=axis_color))),
        yaxis=dict(tickfont=dict(color=axis_color), title=dict(font=dict(color=axis_color)))
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False)
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
    x = None
    if 'target_date' in df.columns:
        x = df['target_date']
    elif 'date' in df.columns:
        x = df['date']
    else:
        x = list(range(len(df)))

    equity = results.get('equity_series', df.get('equity_curve'))
    drawdown = results.get('drawdown_series')

    grid_color = "rgba(148, 163, 184, 0.18)"
    axis_color = "#ffffff"

    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=x, y=equity, mode="lines",
        name="Equity Curve", line=dict(color="#38bdf8", width=3)
    ))
    fig_equity.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        font=dict(color=axis_color),
        xaxis=dict(tickfont=dict(color=axis_color), title=dict(font=dict(color=axis_color))),
        yaxis=dict(tickfont=dict(color=axis_color), title=dict(font=dict(color=axis_color)))
    )
    fig_equity.update_traces(hoverlabel=dict(bgcolor="rgba(15, 23, 42, 0.9)", font_color="#e2e8f0"))
    fig_equity.update_xaxes(showgrid=True, gridcolor=grid_color, zeroline=False, showline=False)
    fig_equity.update_yaxes(showgrid=True, gridcolor=grid_color, zeroline=False, showline=False)

    fig_dd = go.Figure()
    if drawdown is not None:
        fig_dd.add_trace(go.Scatter(
            x=x, y=drawdown, mode="lines",
            name="Drawdown", line=dict(color="#f59e0b", width=2)
        ))
    fig_dd.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=30, b=20),
        hovermode="x unified",
        font=dict(color=axis_color),
        xaxis=dict(tickfont=dict(color=axis_color), title=dict(font=dict(color=axis_color))),
        yaxis=dict(tickfont=dict(color=axis_color), title=dict(font=dict(color=axis_color)))
    )
    fig_dd.update_traces(hoverlabel=dict(bgcolor="rgba(15, 23, 42, 0.9)", font_color="#e2e8f0"))
    fig_dd.update_xaxes(showgrid=True, gridcolor=grid_color, zeroline=False, showline=False)
    fig_dd.update_yaxes(showgrid=True, gridcolor=grid_color, zeroline=True, zerolinecolor="rgba(148, 163, 184, 0.5)")

    return fig_equity, fig_dd
