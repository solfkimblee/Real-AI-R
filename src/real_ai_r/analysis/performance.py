"""绩效分析模块

计算回测结果的各项绩效指标。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_ai_r.engine.backtest import BacktestResult


def calculate_metrics(
    result: BacktestResult,
    risk_free_rate: float = 0.03,
    trading_days: int = 242,
) -> dict:
    """计算完整的绩效指标。

    Parameters
    ----------
    result : BacktestResult
        回测结果。
    risk_free_rate : float
        无风险利率（年化），默认 3%。
    trading_days : int
        年交易日数，默认 242（A股）。

    Returns
    -------
    dict
        绩效指标字典。
    """
    pv = result.portfolio_values
    if len(pv) < 2:
        return {}

    # 日收益率序列
    daily_returns = pv.pct_change().dropna()

    # 总收益率
    total_return = (result.final_capital - result.initial_capital) / result.initial_capital

    # 年化收益率
    n_days = len(pv)
    n_years = n_days / trading_days
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # 年化波动率
    if len(daily_returns) > 0:
        annual_volatility = daily_returns.std() * np.sqrt(trading_days)
    else:
        annual_volatility = 0.0

    # 夏普比率
    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0.0

    # 最大回撤
    cummax = pv.cummax()
    drawdown = (pv - cummax) / cummax
    max_drawdown = drawdown.min()
    max_drawdown_end = drawdown.idxmin()

    # 最大回撤起始日期
    peak_idx = pv[:max_drawdown_end].idxmax() if max_drawdown_end is not None else None

    # 卡尔玛比率 (Calmar Ratio)
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # 胜率
    wins = sum(1 for r in daily_returns if r > 0)
    losses = sum(1 for r in daily_returns if r < 0)
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

    # 盈亏比
    avg_win = daily_returns[daily_returns > 0].mean() if wins > 0 else 0.0
    avg_loss = abs(daily_returns[daily_returns < 0].mean()) if losses > 0 else 0.0
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    # 交易统计
    buy_trades = [t for t in result.trades if t.direction == "BUY"]
    sell_trades = [t for t in result.trades if t.direction == "SELL"]
    total_commission = sum(t.commission for t in result.trades)

    # 索提诺比率 (Sortino Ratio)
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std() * np.sqrt(trading_days)
    else:
        downside_std = 0.0
    sortino_ratio = excess_return / downside_std if downside_std > 0 else 0.0

    metrics = {
        "总收益率": total_return,
        "年化收益率": annual_return,
        "年化波动率": annual_volatility,
        "夏普比率": sharpe_ratio,
        "索提诺比率": sortino_ratio,
        "最大回撤": max_drawdown,
        "最大回撤起始": str(peak_idx) if peak_idx is not None else "N/A",
        "最大回撤结束": str(max_drawdown_end) if max_drawdown_end is not None else "N/A",
        "卡尔玛比率": calmar_ratio,
        "日胜率": win_rate,
        "盈亏比": profit_loss_ratio,
        "交易天数": n_days,
        "买入次数": len(buy_trades),
        "卖出次数": len(sell_trades),
        "总手续费": total_commission,
    }

    result.metrics = metrics
    return metrics


def generate_report_df(metrics: dict) -> pd.DataFrame:
    """将指标字典转为展示用 DataFrame。"""
    format_map = {
        "总收益率": lambda v: f"{v:.2%}",
        "年化收益率": lambda v: f"{v:.2%}",
        "年化波动率": lambda v: f"{v:.2%}",
        "夏普比率": lambda v: f"{v:.4f}",
        "索提诺比率": lambda v: f"{v:.4f}",
        "最大回撤": lambda v: f"{v:.2%}",
        "卡尔玛比率": lambda v: f"{v:.4f}",
        "日胜率": lambda v: f"{v:.2%}",
        "盈亏比": lambda v: f"{v:.4f}",
        "总手续费": lambda v: f"¥{v:,.2f}",
    }

    rows = []
    for k, v in metrics.items():
        formatter = format_map.get(k)
        display_val = formatter(v) if formatter and isinstance(v, (int, float)) else str(v)
        rows.append({"指标": k, "值": display_val})

    return pd.DataFrame(rows)
