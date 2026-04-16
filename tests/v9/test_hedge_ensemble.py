"""Hedge 元学习集成测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_ai_r.v9.combiner.hedge_ensemble import HedgeEnsemble


def test_warmup_returns_equal_weights() -> None:
    h = HedgeEnsemble(members=["a", "b", "c"], warmup=5)
    for _ in range(3):
        h.update({"a": 0.01, "b": 0.0, "c": -0.01})
    w = h.weights()
    for name in ("a", "b", "c"):
        assert abs(w[name] - 1 / 3) < 1e-6


def test_winner_gets_more_weight_post_warmup() -> None:
    h = HedgeEnsemble(members=["w", "l"], warmup=2, eta=50.0, floor=0.0)
    # winner 持续跑赢 10 轮（用大 eta 让收敛明显）
    for _ in range(10):
        h.update({"w": 0.01, "l": -0.01})
    w = h.weights()
    assert w["w"] > w["l"]
    assert w["w"] > 0.9


def test_floor_applied() -> None:
    h = HedgeEnsemble(members=["w", "l"], warmup=1, eta=10.0, floor=0.1)
    for _ in range(20):
        h.update({"w": 0.01, "l": -0.01})
    w = h.weights()
    assert w["l"] >= 0.1 - 1e-6


def test_add_member_dynamically() -> None:
    h = HedgeEnsemble(members=["a"], warmup=0)
    h.update({"a": 0.0})
    h.add_member("b")
    assert "b" in h.weights()


def test_weights_sum_to_one() -> None:
    h = HedgeEnsemble(members=["a", "b", "c"], warmup=0)
    for _ in range(10):
        h.update({"a": 0.01, "b": -0.005, "c": 0.002})
    assert abs(sum(h.weights().values()) - 1.0) < 1e-6


def test_combine_scores_weighted_average() -> None:
    h = HedgeEnsemble(members=["a", "b"], warmup=0, eta=0.0)
    scores = {
        "a": pd.Series([1.0, 2.0, 3.0], index=["x", "y", "z"]),
        "b": pd.Series([3.0, 2.0, 1.0], index=["x", "y", "z"]),
    }
    out = h.combine_scores(scores)
    # 等权时结果应是均值
    assert np.isclose(out.loc["x"], 2.0)
    assert np.isclose(out.loc["y"], 2.0)
    assert np.isclose(out.loc["z"], 2.0)


def test_reset_clears() -> None:
    h = HedgeEnsemble(members=["a", "b"], warmup=0)
    for _ in range(10):
        h.update({"a": 0.01, "b": -0.01})
    h.reset()
    assert h._steps == 0
    # reset 后回到等权
    w = h.weights()
    assert abs(w["a"] - 0.5) < 1e-6
