"""布林带策略

价格触及布林带下轨时买入，触及上轨时卖出。
"""

from __future__ import annotations

import pandas as pd

from real_ai_r.data.indicators import add_bollinger
from real_ai_r.strategies.base import BaseStrategy, Signal


class BollingerStrategy(BaseStrategy):
    """布林带均值回归策略。

    Parameters
    ----------
    window : int
        布林带周期，默认 20。
    num_std : float
        标准差倍数，默认 2.0。
    """

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        super().__init__(name=f"Bollinger({window},{num_std})")
        self.window = window
        self.num_std = num_std

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = data.copy()
        df = add_bollinger(df, window=self.window, num_std=self.num_std)

        signals = pd.Series(Signal.HOLD.value, index=df.index)

        # 价格从下方穿越下轨 → 买入（超卖反弹）
        close_below = df["close"] <= df["bb_lower"]
        prev_above = df["close"].shift(1) > df["bb_lower"].shift(1)
        buy_signal = close_below & prev_above
        # 价格从上方穿越上轨 → 卖出（超买回落）
        close_above = df["close"] >= df["bb_upper"]
        prev_below = df["close"].shift(1) < df["bb_upper"].shift(1)
        sell_signal = close_above & prev_below

        signals[buy_signal] = Signal.BUY.value
        signals[sell_signal] = Signal.SELL.value

        return signals
