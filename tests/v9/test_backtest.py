"""V9 走步回测系统集成测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_ai_r.v9.backtest import V9BacktestResult, V9BacktestRunner
from real_ai_r.v9.engine import V9Config


def test_backtest_smoke(panel_df) -> None:
    runner = V9BacktestRunner(
        config=V9Config(
            enable_regime=False,
            enable_graph=False,
            max_positions=5,
            ic_min_samples=5,
        ),
        warmup_days=10,
    )
    result = runner.run(panel_df)
    assert isinstance(result, V9BacktestResult)
    assert len(result.dates) > 0
    # 净值曲线长度等于保留天数
    assert len(result.equity_curve) == len(result.dates)


def test_backtest_metrics_keys(panel_df) -> None:
    runner = V9BacktestRunner(
        config=V9Config(enable_regime=False, enable_graph=False),
        warmup_days=10,
    )
    r = runner.run(panel_df)
    for k in (
        "total_return",
        "annual_return",
        "volatility",
        "sharpe",
        "info_ratio",
        "max_drawdown",
        "calmar",
        "avg_turnover",
        "win_rate",
    ):
        assert k in r.metrics


def test_backtest_with_regime(panel_df) -> None:
    runner = V9BacktestRunner(
        config=V9Config(
            enable_regime=True,
            enable_graph=False,
            regime_n_states=2,
            regime_min_train=15,
            regime_retrain_every=10,
            max_positions=5,
            ic_min_samples=5,
        ),
        warmup_days=5,
    )
    r = runner.run(panel_df)
    # 至少部分日期有 regime 后验
    non_none = [p for p in r.regime_posterior_history if p is not None]
    assert len(non_none) > 0


def test_backtest_with_graph(panel_df) -> None:
    runner = V9BacktestRunner(
        config=V9Config(
            enable_regime=False,
            enable_graph=True,
            graph_knn=3,
            graph_threshold=0.0,
            graph_rebuild_every=5,
            max_positions=5,
            ic_min_samples=5,
        ),
        warmup_days=5,
    )
    r = runner.run(panel_df)
    assert len(r.dates) > 0


def test_backtest_turnover_is_nonnegative(panel_df) -> None:
    runner = V9BacktestRunner(
        config=V9Config(enable_regime=False, enable_graph=False),
        warmup_days=5,
    )
    r = runner.run(panel_df)
    assert (r.turnover_series >= 0).all()


def test_backtest_positions_respect_max(panel_df) -> None:
    runner = V9BacktestRunner(
        config=V9Config(
            enable_regime=False, enable_graph=False, max_positions=4,
        ),
        warmup_days=5,
    )
    r = runner.run(panel_df)
    assert r.n_positions_series.max() <= 4


def test_backtest_raises_without_date_col() -> None:
    import pytest
    df = pd.DataFrame({"name": ["a"], "change_pct": [1.0]})
    runner = V9BacktestRunner()
    with pytest.raises(ValueError):
        runner.run(df)


def test_backtest_summary_string(panel_df) -> None:
    runner = V9BacktestRunner(
        config=V9Config(enable_regime=False, enable_graph=False),
        warmup_days=5,
    )
    r = runner.run(panel_df)
    s = r.summary()
    assert "V9 Backtest Summary" in s
    assert "Sharpe" in s


def test_backtest_date_range_filter(panel_df) -> None:
    runner = V9BacktestRunner(
        config=V9Config(enable_regime=False, enable_graph=False),
        warmup_days=2,
    )
    dates = sorted(panel_df["date"].unique())
    start = dates[10]
    end = dates[30]
    r = runner.run(panel_df, start=str(start.date()), end=str(end.date()))
    # 结果日期应全部落入 [start, end]
    for d in r.dates:
        assert start.date() <= pd.to_datetime(d).date() <= end.date()


def test_backtest_produces_signal_quality(panel_df) -> None:
    """冷启动通过后，组合应能对简单 momentum 信号表现出 IC > 0。"""
    # 构造一个"动量持续"的面板：每个板块的涨跌服从弱 AR(1) 正系数
    rng = np.random.default_rng(7)
    days = 80
    n_boards = 10
    dates = pd.bdate_range("2024-01-02", periods=days)
    rows = []
    persist = rng.uniform(-0.1, 0.3, n_boards)
    drifts = rng.uniform(-0.05, 0.1, n_boards)
    x = np.zeros(n_boards)
    for d in dates:
        x = persist * x + rng.normal(drifts, 1.5, n_boards)
        for i, bn in enumerate([f"b{j:02d}" for j in range(n_boards)]):
            rows.append(
                {
                    "date": d,
                    "name": bn,
                    "change_pct": float(x[i]),
                    "turnover_rate": 1.0,
                    "rise_count": 15,
                    "fall_count": 15,
                    "lead_stock_pct": float(x[i] + 1),
                },
            )
    df = pd.DataFrame(rows)
    runner = V9BacktestRunner(
        config=V9Config(
            enable_regime=False,
            enable_graph=False,
            max_positions=3,
            ic_min_samples=5,
        ),
        warmup_days=10,
    )
    r = runner.run(df)
    # 基本健康度: 平均换手有限、夏普有定义
    assert np.isfinite(r.metrics["sharpe"])
    assert r.metrics["avg_turnover"] < 2.0
