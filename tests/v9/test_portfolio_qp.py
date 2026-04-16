"""QP 组合优化测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from real_ai_r.v9.optimizer.portfolio_qp import PortfolioOptimizer


@pytest.fixture
def rng():
    return np.random.default_rng(1)


def test_weights_sum_to_one(rng) -> None:
    mu = pd.Series(rng.normal(0, 1, 10), index=[f"b{i}" for i in range(10)])
    opt = PortfolioOptimizer(max_positions=5, risk_aversion=1.0)
    res = opt.optimize(expected_returns=mu)
    assert abs(res.weights.sum() - 1.0) < 1e-4


def test_max_positions_respected(rng) -> None:
    mu = pd.Series(rng.normal(0, 1, 20), index=[f"b{i}" for i in range(20)])
    opt = PortfolioOptimizer(max_positions=5, risk_aversion=1.0)
    res = opt.optimize(expected_returns=mu)
    assert (res.weights > 1e-4).sum() <= 5


def test_max_weight_respected(rng) -> None:
    mu = pd.Series(rng.normal(0, 1, 10), index=[f"b{i}" for i in range(10)])
    opt = PortfolioOptimizer(max_positions=20, max_weight=0.1)
    res = opt.optimize(expected_returns=mu)
    assert res.weights.max() <= 0.1 + 1e-4


def test_forbidden_boards_zero_weight(rng) -> None:
    mu = pd.Series(
        [5.0, 4.0, 3.0, 2.0, 1.0],
        index=["a", "b", "c", "d", "e"],
    )
    opt = PortfolioOptimizer(max_positions=3)
    res = opt.optimize(expected_returns=mu, forbidden=["a", "b"])
    assert res.weights.get("a", 0) < 1e-4
    assert res.weights.get("b", 0) < 1e-4


def test_turnover_penalty_reduces_churn(rng) -> None:
    # 两次 mu 都相同 → 高 turnover_penalty 下权重不应变化
    mu = pd.Series([3.0, 2.0, 1.0, 0.5, 0.1], index=list("abcde"))
    # 第一次优化
    opt = PortfolioOptimizer(max_positions=3, turnover_penalty=10.0)
    res1 = opt.optimize(expected_returns=mu)
    prev_w = {k: float(v) for k, v in res1.weights.items() if v > 1e-6}
    res2 = opt.optimize(expected_returns=mu, prev_weights=prev_w)
    # 换手应很小
    assert res2.turnover < 0.1


def test_empty_input_returns_empty() -> None:
    opt = PortfolioOptimizer()
    res = opt.optimize(expected_returns=pd.Series(dtype=float))
    assert res.status == "EMPTY"
    assert len(res.weights) == 0


def test_optimizer_prefers_high_mu(rng) -> None:
    mu = pd.Series([10.0, -10.0, 0.0], index=["a", "b", "c"])
    opt = PortfolioOptimizer(max_positions=1, risk_aversion=0.0, turnover_penalty=0.0)
    res = opt.optimize(expected_returns=mu)
    # 应选 a
    assert res.weights.loc["a"] > res.weights.loc["b"]
    assert res.weights.loc["a"] > res.weights.loc["c"]


def test_covariance_shrinkage() -> None:
    rng = np.random.default_rng(0)
    Sigma = np.cov(rng.normal(size=(60, 5)).T)
    opt = PortfolioOptimizer(shrinkage=0.5)
    S2 = opt.shrink_cov(Sigma)
    # 收缩后对角线应更接近平均方差
    avg_var = np.trace(Sigma) / 5
    diag_before = np.diag(Sigma)
    diag_after = np.diag(S2)
    var_before = float(np.var(diag_before))
    var_after = float(np.var(diag_after))
    assert var_after <= var_before + 1e-12


def test_large_problem_succeeds(rng) -> None:
    mu = pd.Series(
        rng.normal(0, 1, 50), index=[f"b{i}" for i in range(50)],
    )
    X = rng.normal(0, 1, size=(60, 50))
    opt = PortfolioOptimizer(max_positions=10)
    res = opt.optimize(expected_returns=mu, return_matrix=X)
    assert res.n_positions > 0
    assert res.n_positions <= 10
    assert abs(res.weights.sum() - 1.0) < 1e-3
