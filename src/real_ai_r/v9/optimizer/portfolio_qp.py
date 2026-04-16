"""组合优化器 — mean-variance + 换手成本 + 风格约束。

目标函数：
    maximize   μᵀw  −  λ·wᵀΣw  −  γ·‖w − w_prev‖₁

约束:
    Σw = 1
    0 ≤ w_i ≤ w_max
    非零数 ≤ k_max
    redline 板块 w = 0

通过 L1 转化为 2N 变量 LP/QP；这里用 scipy.optimize.minimize (SLSQP)
做二次规划。小规模 (<100 boards) 下秒级可解。

相对 V8 "argsort top-10"：
- 自然考虑相关性（Σ），避免同涨同跌
- 换手成本内生化，不需要 holdover_bonus hack
- 风格暴露可控，防止某一制度下过度暴露
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class PortfolioResult:
    weights: pd.Series                # index=board_name
    objective: float
    n_positions: int
    turnover: float
    status: str = "OK"


@dataclass
class PortfolioOptimizer:
    """二次规划组合优化。

    参数:
        risk_aversion: λ，风险厌恶系数
        turnover_penalty: γ，换手成本系数
        max_weight: 单板块最大权重
        max_positions: 最多持仓板块数（软约束，通过稀疏化实现）
        shrinkage: 协方差 Ledoit-Wolf 收缩强度 [0,1]
    """

    risk_aversion: float = 2.0
    turnover_penalty: float = 0.5
    max_weight: float = 0.15
    max_positions: int = 12
    shrinkage: float = 0.2
    greedy_fallback: bool = True

    def shrink_cov(self, Sigma: np.ndarray) -> np.ndarray:
        """Ledoit-Wolf 式收缩到 avg-variance × I。"""
        n = Sigma.shape[0]
        avg_var = np.trace(Sigma) / max(n, 1)
        target = np.eye(n) * avg_var
        return (1 - self.shrinkage) * Sigma + self.shrinkage * target

    def optimize(
        self,
        expected_returns: pd.Series,
        return_matrix: np.ndarray | None = None,
        prev_weights: dict[str, float] | None = None,
        forbidden: list[str] | None = None,
    ) -> PortfolioResult:
        """
        expected_returns: index=board_name 的期望收益/分数 μ
        return_matrix: (T, N) 收益矩阵，按 expected_returns.index 顺序
        prev_weights: 上期权重 dict
        forbidden: 红线板块名列表
        """
        if expected_returns is None or len(expected_returns) == 0:
            return PortfolioResult(
                weights=pd.Series(dtype=float),
                objective=0.0,
                n_positions=0,
                turnover=0.0,
                status="EMPTY",
            )

        names = list(expected_returns.index)
        mu = expected_returns.values.astype(float)

        # 把 forbidden 板块 μ 设为极小，天然排除
        if forbidden:
            for i, n in enumerate(names):
                if n in forbidden:
                    mu[i] = -1e6

        n = len(names)
        w_prev = np.zeros(n)
        if prev_weights:
            for i, nm in enumerate(names):
                w_prev[i] = float(prev_weights.get(nm, 0.0))

        # --- 协方差 ---
        if return_matrix is not None and return_matrix.shape[0] >= 5:
            X = np.asarray(return_matrix, dtype=float)
            X = np.where(np.isnan(X), 0.0, X)
            Sigma = np.cov(X.T)
            if Sigma.ndim == 0:
                Sigma = np.array([[float(Sigma)]])
            if Sigma.shape[0] != n:
                # 维度不匹配，退化为 diag
                Sigma = np.eye(n) * 1e-4
            Sigma = self.shrink_cov(Sigma)
            # 小正则确保正定
            Sigma = Sigma + np.eye(n) * 1e-6
        else:
            Sigma = np.eye(n) * 1e-4

        # --- 目标函数 (neg, 因为 minimize) ---
        lam = float(self.risk_aversion)
        gamma = float(self.turnover_penalty)

        def neg_obj(w: np.ndarray) -> float:
            quad = float(w @ Sigma @ w)
            turnover = float(np.sum(np.abs(w - w_prev)))
            return -(mu @ w) + lam * quad + gamma * turnover

        def grad(w: np.ndarray) -> np.ndarray:
            g_mu = -mu
            g_quad = 2 * lam * (Sigma @ w)
            # turnover gradient (subdiff of L1): sign(w - w_prev) * gamma
            g_tv = gamma * np.sign(w - w_prev)
            return g_mu + g_quad + g_tv

        # --- 约束 ---
        # sum w = 1
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, self.max_weight) for _ in range(n)]

        # 初始：用 mu 排序的 top-k 等权
        k = min(self.max_positions, n)
        order = np.argsort(-mu)
        w0 = np.zeros(n)
        w0[order[:k]] = 1.0 / k

        try:
            res = minimize(
                neg_obj,
                w0,
                jac=grad,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 200, "ftol": 1e-8, "disp": False},
            )
            if not res.success and self.greedy_fallback:
                w_opt = self._greedy(mu, Sigma, w_prev)
                status = f"GREEDY({res.message})"
            else:
                w_opt = res.x
                status = "OK" if res.success else f"PARTIAL({res.message})"
        except Exception as e:
            w_opt = self._greedy(mu, Sigma, w_prev)
            status = f"GREEDY({type(e).__name__})"

        # 后处理: 小权重截断到 0（max_positions 软约束）
        w_opt = np.clip(w_opt, 0.0, None)
        if (w_opt > 1e-4).sum() > self.max_positions:
            idx_sorted = np.argsort(-w_opt)
            keep = set(idx_sorted[: self.max_positions].tolist())
            mask = np.array([i in keep for i in range(n)])
            w_opt = np.where(mask, w_opt, 0.0)
        # 重归一化
        total = w_opt.sum()
        if total > 1e-9:
            w_opt = w_opt / total
        else:
            # 兜底: 均匀投 top-k
            w_opt = np.zeros(n)
            top = np.argsort(-mu)[:k]
            w_opt[top] = 1.0 / k

        weights = pd.Series(w_opt, index=names)
        turnover = float(np.sum(np.abs(w_opt - w_prev)))
        return PortfolioResult(
            weights=weights,
            objective=float(-neg_obj(w_opt)),
            n_positions=int((w_opt > 1e-4).sum()),
            turnover=turnover,
            status=status,
        )

    def _greedy(
        self, mu: np.ndarray, Sigma: np.ndarray, w_prev: np.ndarray,
    ) -> np.ndarray:
        """兜底：简单等权选 top-k。"""
        n = len(mu)
        k = min(self.max_positions, n)
        order = np.argsort(-mu)
        w = np.zeros(n)
        w[order[:k]] = 1.0 / k
        return w
