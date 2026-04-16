"""IC 加权与 rank_ic 测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from real_ai_r.v9.combiner.ic_weighter import ICWeighter, rank_ic


def test_rank_ic_perfect_positive() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    r = pd.Series([1, 2, 3, 4, 5])
    assert rank_ic(s, r) == pytest.approx(1.0, abs=1e-10)


def test_rank_ic_perfect_negative() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    r = pd.Series([5, 4, 3, 2, 1])
    assert rank_ic(s, r) == pytest.approx(-1.0, abs=1e-10)


def test_rank_ic_handles_ties() -> None:
    s = pd.Series([1.0, 1.0, 2.0])
    r = pd.Series([1, 2, 3])
    # 不应抛异常
    assert abs(rank_ic(s, r)) <= 1.0 + 1e-9


def test_rank_ic_handles_nan() -> None:
    s = pd.Series([1.0, 2.0, np.nan, 4.0])
    r = pd.Series([1, 2, 3, 4])
    ic = rank_ic(s, r)
    assert np.isfinite(ic)


def test_ic_weighter_warmup_equal_weights() -> None:
    w = ICWeighter(min_samples=10).compute_weights(
        {"f1": [0.1, 0.2], "f2": [0.3, 0.4]},
    )
    # warmup 期等权
    assert abs(w["f1"] - 0.5) < 1e-9
    assert abs(w["f2"] - 0.5) < 1e-9


def test_ic_weighter_icir_assigns_more_to_stable() -> None:
    history = {
        "stable": [0.05] * 20,     # 高均值、低波动
        "noisy": [0.5, -0.5] * 10,  # 均值 0、波动大
    }
    w = ICWeighter(scheme="icir", min_samples=10).compute_weights(history)
    assert w["stable"] > w["noisy"]


def test_ic_weighter_clips_negative_factors() -> None:
    history = {
        "good": [0.1] * 20,
        "bad": [-0.1] * 20,
    }
    w = ICWeighter(scheme="ic_mean", min_samples=10).compute_weights(history)
    assert w["bad"] == 0.0
    assert w["good"] == 1.0


def test_ic_weighter_sums_to_one() -> None:
    history = {
        f"f{i}": list(np.random.default_rng(i).normal(0.05, 0.1, 30))
        for i in range(5)
    }
    w = ICWeighter(scheme="ewma_ic", min_samples=10).compute_weights(history)
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_regime_weights_convexity() -> None:
    ic_by_regime = {
        0: {"f1": [0.3] * 20, "f2": [0.0] * 20},
        1: {"f1": [0.0] * 20, "f2": [0.3] * 20},
    }
    # 制度 0 权重 → f1 应占优
    p0 = np.array([1.0, 0.0])
    w0 = ICWeighter(scheme="icir", min_samples=10).compute_regime_weights(
        ic_by_regime, p0,
    )
    assert w0["f1"] > w0["f2"]
    # 制度 1 权重 → f2 应占优
    p1 = np.array([0.0, 1.0])
    w1 = ICWeighter(scheme="icir", min_samples=10).compute_regime_weights(
        ic_by_regime, p1,
    )
    assert w1["f2"] > w1["f1"]


def test_combine_scores_linear() -> None:
    s1 = pd.Series([1, 2, 3], index=["a", "b", "c"], dtype=float)
    s2 = pd.Series([3, 2, 1], index=["a", "b", "c"], dtype=float)
    out = ICWeighter.combine({"f1": s1, "f2": s2}, {"f1": 0.5, "f2": 0.5})
    assert out.loc["a"] == 2.0
    assert out.loc["b"] == 2.0
    assert out.loc["c"] == 2.0
