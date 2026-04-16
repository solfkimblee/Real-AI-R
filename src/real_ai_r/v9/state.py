"""V9 状态管理 — 有界 deque + 显式 reset + pickle 友好。

核心设计原则：
1. 所有滑窗用 `collections.deque(maxlen=N)` 防止内存泄漏。
2. 显式 `reset()` 接口，回测 / 实盘 / 单测之间隔离。
3. 全部字段 pickle-safe，支持续跑。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class V9State:
    """V9 运行时状态。所有滑窗长度有界。"""

    factor_history_maxlen: int = 120
    board_return_maxlen: int = 120
    portfolio_return_maxlen: int = 120
    regime_feature_maxlen: int = 250

    # 因子 IC 历史: factor_name -> deque[float]
    factor_ic_history: dict[str, deque] = field(default_factory=dict)

    # 板块历史收益: board_name -> deque[float]
    board_return_history: dict[str, deque] = field(default_factory=dict)

    # 组合日超额收益
    portfolio_excess_history: deque = field(default_factory=deque)

    # HMM 训练所需的市场特征 (二维)
    regime_feature_history: deque = field(default_factory=deque)

    # 上期持仓权重 board_name -> weight
    last_weights: dict[str, float] = field(default_factory=dict)

    # 集成成员累积超额
    ensemble_returns: dict[str, deque] = field(default_factory=dict)

    # 每因子的历史截面分 (date -> {board: score})
    factor_score_history: deque = field(default_factory=deque)

    def __post_init__(self) -> None:
        if not isinstance(self.portfolio_excess_history, deque):
            self.portfolio_excess_history = deque(
                list(self.portfolio_excess_history),
                maxlen=self.portfolio_return_maxlen,
            )
        else:
            self.portfolio_excess_history = deque(
                self.portfolio_excess_history, maxlen=self.portfolio_return_maxlen,
            )

        if not isinstance(self.regime_feature_history, deque):
            self.regime_feature_history = deque(
                list(self.regime_feature_history),
                maxlen=self.regime_feature_maxlen,
            )
        else:
            self.regime_feature_history = deque(
                self.regime_feature_history, maxlen=self.regime_feature_maxlen,
            )

        if not isinstance(self.factor_score_history, deque):
            self.factor_score_history = deque(
                list(self.factor_score_history),
                maxlen=self.factor_history_maxlen,
            )
        else:
            self.factor_score_history = deque(
                self.factor_score_history, maxlen=self.factor_history_maxlen,
            )

    # ------------------------------------------------------------------
    # 记录接口
    # ------------------------------------------------------------------

    def record_factor_ic(self, factor_name: str, ic: float) -> None:
        dq = self.factor_ic_history.setdefault(
            factor_name, deque(maxlen=self.factor_history_maxlen),
        )
        dq.append(float(ic))

    def record_board_return(self, board_name: str, ret: float) -> None:
        dq = self.board_return_history.setdefault(
            board_name, deque(maxlen=self.board_return_maxlen),
        )
        dq.append(float(ret))

    def record_portfolio_excess(self, excess: float) -> None:
        self.portfolio_excess_history.append(float(excess))

    def record_regime_feature(self, feat: np.ndarray) -> None:
        self.regime_feature_history.append(np.asarray(feat, dtype=float))

    def record_factor_scores(
        self, factor_scores: dict[str, dict[str, float]],
    ) -> None:
        self.factor_score_history.append(factor_scores)

    def record_ensemble_return(self, member: str, ret: float) -> None:
        dq = self.ensemble_returns.setdefault(
            member, deque(maxlen=self.factor_history_maxlen),
        )
        dq.append(float(ret))

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def get_board_return_matrix(
        self, board_names: list[str], lookback: int,
    ) -> np.ndarray:
        """按 board_names 顺序构造 (lookback, N_boards) 收益矩阵。"""
        rows = []
        for t in range(lookback):
            row = []
            for bn in board_names:
                hist = self.board_return_history.get(bn)
                if hist is None or len(hist) <= t:
                    row.append(np.nan)
                else:
                    # deque 右端为最新
                    idx = len(hist) - lookback + t
                    row.append(hist[idx] if 0 <= idx < len(hist) else np.nan)
            rows.append(row)
        return np.asarray(rows, dtype=float)

    def get_regime_feature_matrix(self) -> np.ndarray | None:
        if len(self.regime_feature_history) == 0:
            return None
        return np.vstack(list(self.regime_feature_history))

    # ------------------------------------------------------------------
    # 重置
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.factor_ic_history.clear()
        self.board_return_history.clear()
        self.portfolio_excess_history.clear()
        self.regime_feature_history.clear()
        self.last_weights.clear()
        self.ensemble_returns.clear()
        self.factor_score_history.clear()

    # ------------------------------------------------------------------
    # pickle 友好
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_ic_history": {
                k: list(v) for k, v in self.factor_ic_history.items()
            },
            "board_return_history": {
                k: list(v) for k, v in self.board_return_history.items()
            },
            "portfolio_excess_history": list(self.portfolio_excess_history),
            "regime_feature_history": [
                x.tolist() for x in self.regime_feature_history
            ],
            "last_weights": dict(self.last_weights),
            "ensemble_returns": {
                k: list(v) for k, v in self.ensemble_returns.items()
            },
            "factor_score_history": list(self.factor_score_history),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> V9State:
        state = cls()
        for k, v in data.get("factor_ic_history", {}).items():
            state.factor_ic_history[k] = deque(
                v, maxlen=state.factor_history_maxlen,
            )
        for k, v in data.get("board_return_history", {}).items():
            state.board_return_history[k] = deque(
                v, maxlen=state.board_return_maxlen,
            )
        state.portfolio_excess_history = deque(
            data.get("portfolio_excess_history", []),
            maxlen=state.portfolio_return_maxlen,
        )
        state.regime_feature_history = deque(
            [np.asarray(x, dtype=float) for x in data.get("regime_feature_history", [])],
            maxlen=state.regime_feature_maxlen,
        )
        state.last_weights = dict(data.get("last_weights", {}))
        for k, v in data.get("ensemble_returns", {}).items():
            state.ensemble_returns[k] = deque(
                v, maxlen=state.factor_history_maxlen,
            )
        state.factor_score_history = deque(
            data.get("factor_score_history", []),
            maxlen=state.factor_history_maxlen,
        )
        return state
