"""V9 因子基础接口 + 注册表。

设计：
- `Factor` 是一个轻量 dataclass，持有名字 + 计算函数
- `FactorContext` 封装额外输入（历史、制度后验等），避免 compute 签名爆炸
- `registry` 是全局单例；子模块 import 时用 `@register` 装饰器自动注册
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd


@dataclass
class FactorContext:
    """因子计算上下文 — 除 board_df 外的所有可选输入。"""

    # 历史收益矩阵 (lookback, N_boards) — 有些因子需要
    board_return_matrix: np.ndarray | None = None

    # 板块名顺序（与 matrix 列对应）
    board_names: list[str] = field(default_factory=list)

    # 当日附加信号 board_name -> value
    extra_signals: dict[str, dict[str, float]] = field(default_factory=dict)

    # 当前日期
    as_of: str | None = None


class FactorFn(Protocol):
    def __call__(
        self, board_df: pd.DataFrame, ctx: FactorContext,
    ) -> pd.Series: ...


@dataclass
class Factor:
    """原子因子。"""

    name: str
    fn: FactorFn
    direction: int = 1  # +1 越大越好，-1 越小越好
    group: str = "misc"

    def compute(
        self, board_df: pd.DataFrame, ctx: FactorContext,
    ) -> pd.Series:
        """执行因子计算并返回 z-score 标准化后的截面分数。"""
        raw = self.fn(board_df, ctx)
        if raw is None or len(raw) == 0:
            return pd.Series(dtype=float)
        raw = pd.to_numeric(raw, errors="coerce")
        raw = raw.replace([np.inf, -np.inf], np.nan)
        # 先用中位数填充 NaN
        med = raw.median()
        if pd.isna(med):
            med = 0.0
        raw = raw.fillna(med)
        # 截面 zscore
        mu = raw.mean()
        sd = raw.std(ddof=0)
        if sd is None or pd.isna(sd) or sd < 1e-9:
            z = pd.Series(np.zeros(len(raw)), index=raw.index)
        else:
            z = (raw - mu) / sd
        # direction
        if self.direction < 0:
            z = -z
        # winsorize 到 [-3, 3]
        return z.clip(lower=-3.0, upper=3.0)


class FactorRegistry:
    """全局因子注册表。"""

    def __init__(self) -> None:
        self._factors: dict[str, Factor] = {}

    def register(
        self, name: str, direction: int = 1, group: str = "misc",
    ) -> Callable[[FactorFn], FactorFn]:
        def deco(fn: FactorFn) -> FactorFn:
            self._factors[name] = Factor(
                name=name, fn=fn, direction=direction, group=group,
            )
            return fn

        return deco

    def get(self, name: str) -> Factor:
        return self._factors[name]

    def names(self) -> list[str]:
        return list(self._factors.keys())

    def all(self) -> list[Factor]:
        return list(self._factors.values())

    def by_group(self, group: str) -> list[Factor]:
        return [f for f in self._factors.values() if f.group == group]


registry = FactorRegistry()
