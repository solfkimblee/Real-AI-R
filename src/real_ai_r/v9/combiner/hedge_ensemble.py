"""Hedge / Exponential Weights 元学习集成。

在线无悔学习：同时跑多个子策略，每日根据实现收益更新每个成员的权重：
    w_k(t+1) ∝ w_k(t) · exp(η · return_k(t))

长期无遗憾地接近事后最优策略。用来软融合 V5/V7/V8/V9 等多版本。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class HedgeEnsemble:
    """在线专家聚合器。

    参数:
        members: 子策略名列表
        eta: 学习率（值越大越快收敛到单一 winner）
        warmup: 前 N 天用等权
        floor: 权重下限（避免某成员永久归零）
        maxlen: 历史收益窗口长度
    """

    members: list[str] = field(default_factory=list)
    eta: float = 2.0
    warmup: int = 10
    floor: float = 0.02
    maxlen: int = 120

    # 累积对数权重
    _log_w: dict[str, float] = field(default_factory=dict)
    # 成员历史收益
    _returns: dict[str, deque] = field(default_factory=dict)
    _steps: int = 0

    def __post_init__(self) -> None:
        for m in self.members:
            self._log_w.setdefault(m, 0.0)
            self._returns.setdefault(m, deque(maxlen=self.maxlen))

    def add_member(self, name: str) -> None:
        if name not in self._log_w:
            self._log_w[name] = 0.0
            self._returns[name] = deque(maxlen=self.maxlen)
            if name not in self.members:
                self.members.append(name)

    def update(self, member_returns: dict[str, float]) -> None:
        """用当日各成员的 realised return 更新权重。"""
        for m, r in member_returns.items():
            if m not in self._log_w:
                self.add_member(m)
            r = float(r)
            self._returns[m].append(r)
            self._log_w[m] += self.eta * r
        self._steps += 1

    def weights(self) -> dict[str, float]:
        """返回归一化权重（warmup 前等权，之后 softmax）。"""
        if not self._log_w:
            return {}
        names = list(self._log_w.keys())
        if self._steps < self.warmup:
            w = 1.0 / len(names)
            return {n: w for n in names}

        lw = np.array([self._log_w[n] for n in names], dtype=float)
        lw = lw - lw.max()
        exp_w = np.exp(lw)
        w = exp_w / (exp_w.sum() + 1e-12)

        # 应用 floor: 预留 N*floor 的地板，剩余按 softmax 分配
        if self.floor > 0:
            N = len(names)
            f = min(self.floor, 1.0 / N)  # 防止 N*floor > 1
            free_mass = 1.0 - N * f
            w = f + free_mass * w
        return {n: float(w[i]) for i, n in enumerate(names)}

    def combine_scores(
        self, member_scores: dict[str, pd.Series],
    ) -> pd.Series:
        """按当前权重聚合多成员的截面分数。"""
        w = self.weights()
        if not w or not member_scores:
            return pd.Series(dtype=float)

        first = next(iter(member_scores.values()))
        if first is None or len(first) == 0:
            return pd.Series(dtype=float)
        idx = first.index
        out = pd.Series(np.zeros(len(idx)), index=idx)
        for m, s in member_scores.items():
            if m not in w or s is None or len(s) == 0:
                continue
            out = out.add(
                s.reindex(idx).fillna(0.0) * w[m], fill_value=0.0,
            )
        return out

    def reset(self) -> None:
        for m in self._log_w:
            self._log_w[m] = 0.0
            self._returns[m].clear()
        self._steps = 0
