"""单元测试 — 回测引擎"""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_ai_r.analysis.performance import calculate_metrics
from real_ai_r.engine.backtest import BacktestEngine, BacktestResult
from real_ai_r.strategies.ma_cross import MACrossStrategy


def _make_sample_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """生成模拟行情数据。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 10 + np.cumsum(rng.randn(n) * 0.3)
    close = np.maximum(close, 1.0)
    high = close + rng.rand(n) * 0.5
    low = close - rng.rand(n) * 0.5
    low = np.maximum(low, 0.5)
    open_ = close + rng.randn(n) * 0.2
    volume = rng.randint(100000, 1000000, size=n)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class TestBacktestEngine:
    def test_basic_run(self) -> None:
        data = _make_sample_data()
        strategy = MACrossStrategy(short_window=5, long_window=20)
        engine = BacktestEngine(data=data, strategy=strategy, initial_capital=100000)
        result = engine.run()

        assert isinstance(result, BacktestResult)
        assert result.initial_capital == 100000
        assert result.final_capital > 0
        assert len(result.portfolio_values) == len(data)

    def test_portfolio_starts_at_initial_capital(self) -> None:
        data = _make_sample_data()
        strategy = MACrossStrategy(short_window=5, long_window=20)
        engine = BacktestEngine(data=data, strategy=strategy, initial_capital=50000)
        result = engine.run()

        # 第一天净值应该等于初始资金（还没有交易）
        assert abs(result.portfolio_values.iloc[0] - 50000) < 1.0

    def test_trades_are_recorded(self) -> None:
        data = _make_sample_data(n=300)
        strategy = MACrossStrategy(short_window=5, long_window=20)
        engine = BacktestEngine(data=data, strategy=strategy)
        result = engine.run()

        # 应该有交易记录
        assert len(result.trades) > 0
        for t in result.trades:
            assert t.direction in ("BUY", "SELL")
            assert t.price > 0
            assert t.shares > 0

    def test_commission_deducted(self) -> None:
        data = _make_sample_data(n=300)
        strategy = MACrossStrategy(short_window=5, long_window=20)
        engine = BacktestEngine(
            data=data,
            strategy=strategy,
            initial_capital=100000,
            commission_rate=0.001,  # 千一
        )
        result = engine.run()

        total_commission = sum(t.commission for t in result.trades)
        assert total_commission > 0

    def test_t_plus_1_rule(self) -> None:
        """验证 T+1 规则：买入当天不能卖出。"""
        data = _make_sample_data(n=300)
        strategy = MACrossStrategy(short_window=5, long_window=20)
        engine = BacktestEngine(data=data, strategy=strategy)
        result = engine.run()

        # 检查没有同日买卖
        buy_dates = {t.date for t in result.trades if t.direction == "BUY"}
        sell_dates = {t.date for t in result.trades if t.direction == "SELL"}
        assert buy_dates.isdisjoint(sell_dates), "T+1: 同日不应有买卖"

    def test_summary(self) -> None:
        data = _make_sample_data()
        strategy = MACrossStrategy(short_window=5, long_window=20)
        engine = BacktestEngine(data=data, strategy=strategy)
        result = engine.run()
        summary = result.summary()
        assert "策略" in summary
        assert "初始资金" in summary


class TestPerformanceMetrics:
    def test_metrics_calculated(self) -> None:
        data = _make_sample_data(n=300)
        strategy = MACrossStrategy(short_window=5, long_window=20)
        engine = BacktestEngine(data=data, strategy=strategy)
        result = engine.run()
        metrics = calculate_metrics(result)

        assert "总收益率" in metrics
        assert "年化收益率" in metrics
        assert "夏普比率" in metrics
        assert "最大回撤" in metrics
        assert "日胜率" in metrics

    def test_max_drawdown_is_negative(self) -> None:
        data = _make_sample_data(n=300)
        strategy = MACrossStrategy(short_window=5, long_window=20)
        engine = BacktestEngine(data=data, strategy=strategy)
        result = engine.run()
        metrics = calculate_metrics(result)

        assert metrics["最大回撤"] <= 0

    def test_win_rate_in_range(self) -> None:
        data = _make_sample_data(n=300)
        strategy = MACrossStrategy(short_window=5, long_window=20)
        engine = BacktestEngine(data=data, strategy=strategy)
        result = engine.run()
        metrics = calculate_metrics(result)

        assert 0 <= metrics["日胜率"] <= 1
