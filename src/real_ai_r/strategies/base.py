"""策略基类 — 所有策略的抽象接口

所有自定义策略必须继承 BaseStrategy 并实现 generate_signals 方法。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Signal(Enum):
    """交易信号枚举。"""

    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class StrategyConfig:
    """策略配置基类。"""

    name: str = "base"
    params: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    """策略抽象基类。

    子类需要实现：
    - generate_signals(data): 根据行情数据生成交易信号序列

    信号约定：
    - Signal.BUY  (1) : 买入
    - Signal.SELL (-1): 卖出
    - Signal.HOLD (0) : 持有/观望
    """

    def __init__(self, name: str = "base") -> None:
        self.name = name

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号序列。

        Parameters
        ----------
        data : pd.DataFrame
            包含 OHLCV 数据及可能的技术指标。

        Returns
        -------
        pd.Series
            与 data 等长的信号序列，值为 Signal 枚举值的 .value（1, -1, 0）。
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
