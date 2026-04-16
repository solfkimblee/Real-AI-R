"""V9 共享 fixtures — 合成市场数据用于单元 / 集成测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_board_df(rng):
    """单日截面 DataFrame，10 个板块。"""
    n = 10
    names = [f"板块{i:02d}" for i in range(n)]
    return pd.DataFrame(
        {
            "name": names,
            "code": [f"BK{i:04d}" for i in range(n)],
            "change_pct": rng.normal(0, 2.0, n),
            "turnover_rate": rng.uniform(0.5, 5.0, n),
            "rise_count": rng.integers(5, 30, n),
            "fall_count": rng.integers(5, 30, n),
            "lead_stock": [f"股{i}" for i in range(n)],
            "lead_stock_pct": rng.uniform(-5, 10, n),
        },
    )


@pytest.fixture
def panel_df(rng):
    """面板数据 — 60 天 × 15 板块。用于回测。"""
    days = 60
    n_boards = 15
    dates = pd.bdate_range("2024-01-02", periods=days)
    board_names = [f"板块{i:02d}" for i in range(n_boards)]

    # 为每个板块生成 AR(1) 式涨跌
    rows = []
    for bi, bn in enumerate(board_names):
        # 每个板块有略微不同的漂移和波动
        drift = rng.uniform(-0.05, 0.15)
        vol = rng.uniform(1.5, 3.0)
        x = 0.0
        for d in dates:
            x = 0.3 * x + rng.normal(drift, vol)
            rise = rng.integers(5, 30)
            fall = rng.integers(5, 30)
            rows.append(
                {
                    "date": d,
                    "name": bn,
                    "code": f"BK{bi:04d}",
                    "change_pct": float(x),
                    "turnover_rate": float(rng.uniform(0.5, 5.0)),
                    "rise_count": int(rise),
                    "fall_count": int(fall),
                    "lead_stock_pct": float(rng.uniform(-5, 10)),
                },
            )
    return pd.DataFrame(rows)


@pytest.fixture
def return_matrix(rng):
    """(T=60, N=15) 收益矩阵。"""
    return rng.normal(0, 2.0, size=(60, 15)).astype(float)
