"""可视化模块 — Plotly 图表

提供回测结果的交互式可视化。
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from real_ai_r.engine.backtest import BacktestResult


def plot_portfolio_value(result: BacktestResult, benchmark: dict | None = None) -> go.Figure:
    """绘制组合净值曲线。

    Parameters
    ----------
    result : BacktestResult
        回测结果。
    benchmark : dict | None
        基准数据 {"name": str, "values": pd.Series}。
    """
    fig = go.Figure()

    # 归一化净值
    pv = result.portfolio_values
    norm_pv = pv / pv.iloc[0]

    fig.add_trace(
        go.Scatter(
            x=norm_pv.index,
            y=norm_pv.values,
            mode="lines",
            name=result.strategy_name,
            line=dict(color="#e74c3c", width=2),
        )
    )

    if benchmark:
        bm = benchmark["values"]
        norm_bm = bm / bm.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=norm_bm.index,
                y=norm_bm.values,
                mode="lines",
                name=benchmark["name"],
                line=dict(color="#95a5a6", width=1.5, dash="dash"),
            )
        )

    # 标记买卖点
    for trade in result.trades:
        color = "#2ecc71" if trade.direction == "BUY" else "#e74c3c"
        symbol = "triangle-up" if trade.direction == "BUY" else "triangle-down"
        # 找到对应日期的归一化净值
        if trade.date in norm_pv.index:
            y_val = norm_pv[trade.date]
            fig.add_trace(
                go.Scatter(
                    x=[trade.date],
                    y=[y_val],
                    mode="markers",
                    marker=dict(color=color, size=10, symbol=symbol),
                    name=trade.direction,
                    showlegend=False,
                    hovertext=f"{trade.direction} {trade.shares}股 @ ¥{trade.price:.2f}",
                )
            )

    fig.update_layout(
        title=f"组合净值曲线 — {result.strategy_name}",
        xaxis_title="日期",
        yaxis_title="归一化净值",
        template="plotly_white",
        hovermode="x unified",
        height=500,
    )

    return fig


def plot_drawdown(result: BacktestResult) -> go.Figure:
    """绘制回撤曲线。"""
    pv = result.portfolio_values
    cummax = pv.cummax()
    drawdown = (pv - cummax) / cummax

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill="tozeroy",
            mode="lines",
            name="回撤",
            line=dict(color="#e74c3c", width=1),
            fillcolor="rgba(231, 76, 60, 0.3)",
        )
    )

    fig.update_layout(
        title="回撤曲线",
        xaxis_title="日期",
        yaxis_title="回撤",
        yaxis_tickformat=".1%",
        template="plotly_white",
        height=300,
    )

    return fig


def plot_candlestick_with_signals(
    data: dict,
    result: BacktestResult,
) -> go.Figure:
    """绘制K线图 + 交易信号。

    Parameters
    ----------
    data : dict
        行情数据，需包含 date, open, high, low, close, volume。
    result : BacktestResult
        回测结果（用于标记买卖点）。
    """
    df = data if hasattr(data, "iloc") else data
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
    )

    # K线
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
            increasing_line_color="#e74c3c",
            decreasing_line_color="#2ecc71",
        ),
        row=1,
        col=1,
    )

    # 成交量
    colors = [
        "#e74c3c" if df["close"].iloc[i] >= df["open"].iloc[i] else "#2ecc71"
        for i in range(len(df))
    ]
    fig.add_trace(
        go.Bar(x=df["date"], y=df["volume"], name="成交量", marker_color=colors, opacity=0.5),
        row=2,
        col=1,
    )

    # 买卖标记
    for trade in result.trades:
        color = "#ff6600" if trade.direction == "BUY" else "#0066ff"
        symbol = "triangle-up" if trade.direction == "BUY" else "triangle-down"
        fig.add_trace(
            go.Scatter(
                x=[trade.date],
                y=[trade.price],
                mode="markers",
                marker=dict(color=color, size=12, symbol=symbol),
                name=trade.direction,
                showlegend=False,
                hovertext=f"{trade.direction} {trade.shares}股 @ ¥{trade.price:.2f}",
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        title=f"K线图 + 交易信号 — {result.strategy_name}",
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        height=700,
    )

    return fig


def plot_monthly_returns(result: BacktestResult) -> go.Figure:
    """绘制月度收益热力图。"""
    pv = result.portfolio_values
    daily_returns = pv.pct_change().dropna()

    # 按月聚合
    monthly = daily_returns.groupby([daily_returns.index.year, daily_returns.index.month]).apply(
        lambda x: (1 + x).prod() - 1
    )

    if monthly.empty:
        fig = go.Figure()
        fig.update_layout(title="月度收益（数据不足）")
        return fig

    # 构建矩阵
    years = sorted(set(idx[0] for idx in monthly.index))
    months = list(range(1, 13))
    z = []
    for year in years:
        row = []
        for month in months:
            if (year, month) in monthly.index:
                row.append(monthly[(year, month)] * 100)
            else:
                row.append(None)
        z.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[f"{m}月" for m in months],
            y=[str(y) for y in years],
            colorscale="RdYlGn",
            text=[[f"{v:.1f}%" if v is not None else "" for v in row] for row in z],
            texttemplate="%{text}",
            hovertemplate="年份: %{y}<br>月份: %{x}<br>收益: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title="月度收益热力图",
        template="plotly_white",
        height=300,
    )

    return fig
