"""板块相关性图传播 — 从板块历史收益构建相关图，对因子分数做传播。

核心思想：
    score' = (1-α)·score + α·D⁻¹·W·score

其中：
    W = 板块对板块相关性（可选阈值化/KNN 稀疏化）
    D = W 的行和（度矩阵）
    α = 传播系数

迭代 T 步等价于做 personalized PageRank。
一个板块热起来，相关板块自动沾光 → 形成主题聚焦，
解决 V8 独立打分不捕捉联动的硬伤。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def build_correlation_graph(
    return_matrix: np.ndarray,
    board_names: list[str],
    knn: int = 10,
    threshold: float = 0.3,
    clip_negative: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """从历史收益构建板块相关性邻接矩阵。

    参数:
        return_matrix: (T, N) 板块日收益矩阵
        board_names: 长度 N 的板块名
        knn: 每个节点保留 top-k 相关邻居（稀疏化）
        threshold: 相关系数阈值，低于则置 0
        clip_negative: 是否把负相关置零（只传播同向主题）

    返回:
        (W, names) 其中 W 是 (N, N) 对称邻接矩阵（对角 0）
    """
    if return_matrix is None or len(return_matrix) < 5:
        n = len(board_names)
        return np.zeros((n, n)), board_names

    X = np.asarray(return_matrix, dtype=float)
    # 处理 NaN
    X = np.where(np.isnan(X), 0.0, X)

    # 相关矩阵
    Xc = X - X.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(Xc**2, axis=0)) + 1e-12
    Xn = Xc / denom
    C = Xn.T @ Xn  # (N, N)
    np.fill_diagonal(C, 0.0)

    if clip_negative:
        C = np.clip(C, 0.0, None)

    # 阈值化
    C = np.where(np.abs(C) < threshold, 0.0, C)

    # KNN 稀疏化
    if knn is not None and knn > 0 and knn < C.shape[0]:
        W = np.zeros_like(C)
        for i in range(C.shape[0]):
            idx = np.argsort(-np.abs(C[i]))[:knn]
            W[i, idx] = C[i, idx]
        # 对称化
        W = np.maximum(W, W.T)
    else:
        W = C

    return W, board_names


@dataclass
class GraphPropagator:
    """标签传播 / personalized PageRank 式的分数传播。

    参数:
        alpha: 传播系数（0 = 无传播，1 = 全传播）
        n_steps: 迭代步数
        normalize_row: 是否按行归一化（归一化 → 传播不放大总量）
    """

    alpha: float = 0.25
    n_steps: int = 2
    normalize_row: bool = True

    def propagate(
        self, scores: pd.Series, W: np.ndarray, board_names: list[str],
    ) -> pd.Series:
        """对 scores（index=board_name）做传播。"""
        if W is None or W.size == 0 or len(board_names) == 0:
            return scores.copy()

        s = scores.reindex(board_names).fillna(0.0).values.astype(float)

        # 度矩阵归一化
        W_work = W.copy()
        if self.normalize_row:
            row_sum = W_work.sum(axis=1, keepdims=True) + 1e-12
            W_norm = W_work / row_sum
        else:
            W_norm = W_work

        out = s.copy()
        for _ in range(self.n_steps):
            out = (1.0 - self.alpha) * s + self.alpha * (W_norm @ out)

        result = pd.Series(out, index=board_names)
        # 把原始 index 里但不在 board_names 里的保留原值
        missing = scores.index.difference(board_names)
        if len(missing) > 0:
            result = pd.concat([result, scores.loc[missing]])
        return result.reindex(scores.index)
