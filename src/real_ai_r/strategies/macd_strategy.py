"""MACD 策略

基于 MACD 金叉/死叉信号进行交易。
"""

from __future__ import annotations

import pandas as pd

from real_ai_r.data.indicators import add_macd
from real_ai_r.strategies.base import BaseStrategy, Signal


class MACDStrategy(BaseStrategy):
    """MACD 金叉/死叉策略。

    Parameters
    ----------
    fast : int
        快线周期，默认 12。
    slow : int
        慢线周期，默认 26。
    signal_period : int
        信号线周期，默认 9。
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal_period: int = 9) -> None:
        super().__init__(name=f"MACD({fast},{slow},{signal_period})")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = data.copy()
        df = add_macd(df, fast=self.fast, slow=self.slow, signal=self.signal_period)

        signals = pd.Series(Signal.HOLD.value, index=df.index)

        # MACD 上穿信号线（金叉）→ 买入
        golden_cross = (df["macd"] > df["macd_signal"]) & (
            df["macd"].shift(1) <= df["macd_signal"].shift(1)
        )
        # MACD 下穿信号线（死叉）→ 卖出
        death_cross = (df["macd"] < df["macd_signal"]) & (
            df["macd"].shift(1) >= df["macd_signal"].shift(1)
        )

        signals[golden_cross] = Signal.BUY.value
        signals[death_cross] = Signal.SELL.value

        return signals
