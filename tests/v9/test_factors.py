"""V9 因子测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_ai_r.v9.factors import registry
from real_ai_r.v9.factors.base import FactorContext


def test_registry_has_core_factors() -> None:
    names = registry.names()
    for f in (
        "momentum_1d",
        "momentum_5d",
        "momentum_20d",
        "reversal_1d",
        "volatility_20d",
        "turnover",
        "rise_ratio",
        "lead_strength",
        "dispersion",
        "main_inflow",
        "catalyst_score",
        "redline_block",
    ):
        assert f in names, f"missing factor: {f}"


def test_momentum_1d_correct_direction(small_board_df) -> None:
    f = registry.get("momentum_1d")
    ctx = FactorContext(board_names=list(small_board_df["name"]))
    z = f.compute(small_board_df, ctx)
    # 最大 change_pct 对应的 name 应该 z 值最大
    top_name = small_board_df.sort_values(
        "change_pct", ascending=False,
    ).iloc[0]["name"]
    assert z.loc[top_name] == z.max()


def test_reversal_1d_is_negated(small_board_df) -> None:
    """reversal direction=-1 → 应与 momentum_1d 反号。"""
    mom = registry.get("momentum_1d").compute(
        small_board_df, FactorContext(board_names=list(small_board_df["name"])),
    )
    rev = registry.get("reversal_1d").compute(
        small_board_df, FactorContext(board_names=list(small_board_df["name"])),
    )
    diff = (mom + rev).abs().max()
    assert diff < 1e-9


def test_zscore_is_normalized(small_board_df) -> None:
    f = registry.get("momentum_1d")
    ctx = FactorContext(board_names=list(small_board_df["name"]))
    z = f.compute(small_board_df, ctx)
    assert abs(z.mean()) < 1e-6
    assert abs(z.std(ddof=0) - 1.0) < 0.5  # clip 会影响


def test_rise_ratio_bounded(small_board_df) -> None:
    f = registry.get("rise_ratio")
    ctx = FactorContext(board_names=list(small_board_df["name"]))
    z = f.compute(small_board_df, ctx)
    assert len(z) == len(small_board_df)
    # 应在 [-3, 3] 范围内（clip 后）
    assert z.max() <= 3.0 and z.min() >= -3.0


def test_momentum_5d_uses_column_if_present() -> None:
    df = pd.DataFrame(
        {
            "name": ["A", "B", "C"],
            "change_pct": [1.0, 2.0, 3.0],
            "momentum_5d": [10.0, 5.0, 1.0],
        },
    )
    ctx = FactorContext(board_names=["A", "B", "C"])
    z = registry.get("momentum_5d").compute(df, ctx)
    # A 最大 → z 最大
    assert z.loc["A"] == z.max()


def test_momentum_5d_falls_back_to_matrix() -> None:
    df = pd.DataFrame({"name": ["A", "B", "C"], "change_pct": [0, 0, 0]})
    mat = np.array(
        [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ],
        dtype=float,
    )
    ctx = FactorContext(board_return_matrix=mat, board_names=["A", "B", "C"])
    z = registry.get("momentum_5d").compute(df, ctx)
    # C 累加最大 → z 最大
    assert z.loc["C"] == z.max()


def test_volatility_direction_low_is_better() -> None:
    df = pd.DataFrame({"name": ["A", "B", "C"], "change_pct": [0, 0, 0]})
    # A 波动小、B 中、C 大
    rng = np.random.default_rng(0)
    mat = np.stack(
        [
            rng.normal(0, 0.1, 30),
            rng.normal(0, 1.0, 30),
            rng.normal(0, 5.0, 30),
        ],
        axis=1,
    )
    ctx = FactorContext(board_return_matrix=mat, board_names=["A", "B", "C"])
    z = registry.get("volatility_20d").compute(df, ctx)
    # direction=-1 → A (最低波动) 应得最高分
    assert z.loc["A"] > z.loc["C"]


def test_redline_block_marks_forbidden() -> None:
    df = pd.DataFrame({"name": ["A", "B"]})
    extra = {"A": {"redline": True}, "B": {"redline": False}}
    ctx = FactorContext(board_names=["A", "B"], extra_signals=extra)
    z = registry.get("redline_block").compute(df, ctx)
    # direction=-1: redline=True 的会被负号 → A 分数应低于 B
    assert z.loc["A"] < z.loc["B"]


def test_catalyst_score_reads_extra_signals() -> None:
    df = pd.DataFrame({"name": ["A", "B"]})
    extra = {"A": {"catalyst_score": 5.0}, "B": {"catalyst_score": -2.0}}
    ctx = FactorContext(board_names=["A", "B"], extra_signals=extra)
    z = registry.get("catalyst_score").compute(df, ctx)
    assert z.loc["A"] > z.loc["B"]


def test_factor_handles_empty() -> None:
    df = pd.DataFrame({"name": []})
    ctx = FactorContext(board_names=[])
    z = registry.get("momentum_1d").compute(df, ctx)
    assert len(z) == 0


def test_factor_handles_all_nan() -> None:
    df = pd.DataFrame(
        {"name": ["A", "B"], "change_pct": [np.nan, np.nan]},
    )
    ctx = FactorContext(board_names=["A", "B"])
    z = registry.get("momentum_1d").compute(df, ctx)
    # 不应该抛异常
    assert len(z) == 2
    assert (z.fillna(0.0) == 0.0).all()
