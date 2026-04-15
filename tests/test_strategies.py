"""单元测试 — 策略模块"""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_ai_r.strategies.base import Signal
from real_ai_r.strategies.bollinger_strategy import BollingerStrategy
from real_ai_r.strategies.ma_cross import MACrossStrategy
from real_ai_r.strategies.macd_strategy import MACDStrategy


def _make_sample_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """生成模拟行情数据。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 10 + np.cumsum(rng.randn(n) * 0.3)
    close = np.maximum(close, 1.0)  # 确保价格为正
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


class TestMACrossStrategy:
    def test_signal_length(self) -> None:
        data = _make_sample_data()
        strategy = MACrossStrategy(short_window=5, long_window=20)
        signals = strategy.generate_signals(data)
        assert len(signals) == len(data)

    def test_signal_values(self) -> None:
        data = _make_sample_data()
        strategy = MACrossStrategy(short_window=5, long_window=20)
        signals = strategy.generate_signals(data)
        valid_values = {Signal.BUY.value, Signal.SELL.value, Signal.HOLD.value}
        assert set(signals.unique()).issubset(valid_values)

    def test_has_buy_and_sell(self) -> None:
        data = _make_sample_data(n=300)
        strategy = MACrossStrategy(short_window=5, long_window=20)
        signals = strategy.generate_signals(data)
        # 在足够长的随机数据中应该有买卖信号
        assert Signal.BUY.value in signals.values
        assert Signal.SELL.value in signals.values

    def test_repr(self) -> None:
        strategy = MACrossStrategy(short_window=5, long_window=20)
        assert "MA_Cross" in repr(strategy)


class TestMACDStrategy:
    def test_signal_length(self) -> None:
        data = _make_sample_data()
        strategy = MACDStrategy()
        signals = strategy.generate_signals(data)
        assert len(signals) == len(data)

    def test_signal_values(self) -> None:
        data = _make_sample_data()
        strategy = MACDStrategy()
        signals = strategy.generate_signals(data)
        valid_values = {Signal.BUY.value, Signal.SELL.value, Signal.HOLD.value}
        assert set(signals.unique()).issubset(valid_values)


class TestBollingerStrategy:
    def test_signal_length(self) -> None:
        data = _make_sample_data()
        strategy = BollingerStrategy()
        signals = strategy.generate_signals(data)
        assert len(signals) == len(data)

    def test_signal_values(self) -> None:
        data = _make_sample_data()
        strategy = BollingerStrategy()
        signals = strategy.generate_signals(data)
        valid_values = {Signal.BUY.value, Signal.SELL.value, Signal.HOLD.value}
        assert set(signals.unique()).issubset(valid_values)
