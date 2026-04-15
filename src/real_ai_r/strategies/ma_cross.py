"""双均线交叉策略

经典趋势跟踪策略：短期均线上穿长期均线时买入，下穿时卖出。
"""

from __future__ import annotations

import pandas as pd

from real_ai_r.data.indicators import add_ma
from real_ai_r.strategies.base import BaseStrategy, Signal


class MACrossStrategy(BaseStrategy):
    """双均线交叉策略。

    Parameters
    ----------
    short_window : int
        短期均线周期，默认 5。
    long_window : int
        长期均线周期，默认 20。
    """

    def __init__(self, short_window: int = 5, long_window: int = 20) -> None:
        super().__init__(name=f"MA_Cross({short_window},{long_window})")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        df = data.copy()
        df = add_ma(df, window=self.short_window)
        df = add_ma(df, window=self.long_window)

        short_col = f"ma_{self.short_window}"
        long_col = f"ma_{self.long_window}"

        signals = pd.Series(Signal.HOLD.value, index=df.index)

        # 短均线上穿长均线 → 买入
        cross_up = (df[short_col] > df[long_col]) & (
            df[short_col].shift(1) <= df[long_col].shift(1)
        )
        # 短均线下穿长均线 → 卖出
        cross_down = (df[short_col] < df[long_col]) & (
            df[short_col].shift(1) >= df[long_col].shift(1)
        )

        signals[cross_up] = Signal.BUY.value
        signals[cross_down] = Signal.SELL.value

        return signals
