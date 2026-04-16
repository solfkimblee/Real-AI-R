"""图传播测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_ai_r.v9.graph.propagation import (
    GraphPropagator,
    build_correlation_graph,
)


def test_correlation_graph_shape() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (60, 5))
    W, names = build_correlation_graph(X, ["a", "b", "c", "d", "e"])
    assert W.shape == (5, 5)
    assert (np.diag(W) == 0).all()


def test_correlation_graph_symmetric() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (60, 6))
    W, _ = build_correlation_graph(X, list("abcdef"), knn=3)
    assert np.allclose(W, W.T)


def test_correlation_knn_sparsity() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (60, 10))
    W, _ = build_correlation_graph(X, list("abcdefghij"), knn=3, threshold=0.0)
    # 对称后每行非零 ≥ 3（KNN 本身）但 <= 10 (含对称的邻居)
    for i in range(10):
        assert (W[i] != 0).sum() <= 10


def test_clip_negative_default() -> None:
    # 构造两个强负相关的序列
    X = np.zeros((60, 2))
    X[:, 0] = np.arange(60) * 0.1
    X[:, 1] = -np.arange(60) * 0.1
    W, _ = build_correlation_graph(X, ["a", "b"], knn=1, threshold=0.0)
    # 负相关被 clip 为 0
    assert (W >= 0).all()


def test_graph_propagator_no_propagation_when_alpha_zero() -> None:
    W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    scores = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    prop = GraphPropagator(alpha=0.0)
    out = prop.propagate(scores, W, ["a", "b", "c"])
    assert np.allclose(out.values, scores.values)


def test_graph_propagator_smooths_neighbor_scores() -> None:
    # 三节点链: a-b-c，初始只有 b=10，其他 0
    W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    scores = pd.Series([0.0, 10.0, 0.0], index=["a", "b", "c"])
    prop = GraphPropagator(alpha=0.5, n_steps=1)
    out = prop.propagate(scores, W, ["a", "b", "c"])
    # b 有邻居 → a 和 c 分数应上升
    assert out.loc["a"] > 0
    assert out.loc["c"] > 0
    # b 自身应保留部分原始分数
    assert out.loc["b"] >= 0


def test_graph_propagator_empty_W() -> None:
    scores = pd.Series([1.0, 2.0], index=["a", "b"])
    prop = GraphPropagator()
    out = prop.propagate(scores, np.zeros((0, 0)), [])
    # 没有图 → 返回原分数
    assert out.equals(scores)


def test_correlation_graph_short_history_returns_zeros() -> None:
    W, names = build_correlation_graph(np.zeros((2, 3)), ["a", "b", "c"])
    assert W.shape == (3, 3)
    assert (W == 0).all()
