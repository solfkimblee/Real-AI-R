"""价量类因子。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_ai_r.v9.factors.base import FactorContext, registry


def _series_from_df(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df["name"])
    return pd.to_numeric(df[col], errors="coerce").set_axis(df["name"].values)


@registry.register("momentum_1d", direction=1, group="price")
def momentum_1d(board_df: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """当日涨跌幅。"""
    return _series_from_df(board_df, "change_pct")


@registry.register("momentum_5d", direction=1, group="price")
def momentum_5d(board_df: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """5 日动量，优先用 board_df 列，否则从历史矩阵聚合。"""
    if "momentum_5d" in board_df.columns:
        return _series_from_df(board_df, "momentum_5d")
    mat = ctx.board_return_matrix
    if mat is None or len(ctx.board_names) == 0:
        return pd.Series(dtype=float)
    k = min(5, mat.shape[0])
    mom = np.nansum(mat[-k:, :], axis=0)
    s = pd.Series(mom, index=ctx.board_names)
    return s.reindex(board_df["name"].values).fillna(0.0)


@registry.register("momentum_20d", direction=1, group="price")
def momentum_20d(board_df: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """20 日动量。"""
    mat = ctx.board_return_matrix
    if mat is None or len(ctx.board_names) == 0:
        return pd.Series(dtype=float)
    k = min(20, mat.shape[0])
    mom = np.nansum(mat[-k:, :], axis=0)
    s = pd.Series(mom, index=ctx.board_names)
    return s.reindex(board_df["name"].values).fillna(0.0)


@registry.register("reversal_1d", direction=-1, group="price")
def reversal_1d(board_df: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """1 日反转（direction=-1: 跌得多的下期反弹）。"""
    return _series_from_df(board_df, "change_pct")


@registry.register("volatility_20d", direction=-1, group="price")
def volatility_20d(board_df: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """20 日收益波动率（越低越好）。"""
    mat = ctx.board_return_matrix
    if mat is None or mat.shape[0] < 5:
        return pd.Series(np.zeros(len(board_df)), index=board_df["name"].values)
    k = min(20, mat.shape[0])
    window = mat[-k:, :]
    # 若某列全为 NaN（新板块）会触发 warning；用 errstate 静默
    with np.errstate(all="ignore"):
        vol = np.nanstd(window, axis=0, ddof=0)
    vol = np.where(np.isnan(vol), 0.0, vol)
    s = pd.Series(vol, index=ctx.board_names)
    return s.reindex(board_df["name"].values).fillna(s.median() if len(s) > 0 else 0.0)


@registry.register("turnover", direction=1, group="price")
def turnover(board_df: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """换手率。"""
    return _series_from_df(board_df, "turnover_rate")


@registry.register("turnover_surge", direction=1, group="price")
def turnover_surge(board_df: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """当日换手率相对近期均值的溢出（反映资金突然进入）。"""
    cur = pd.to_numeric(
        board_df.get("turnover_rate", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0).values
    names = board_df["name"].values
    # 如果 extra_signals 提供了 turnover_ma，用它
    ma = np.array(
        [
            ctx.extra_signals.get(n, {}).get("turnover_ma20", np.nan)
            for n in names
        ],
        dtype=float,
    )
    if np.isnan(ma).all():
        return pd.Series(np.zeros(len(names)), index=names)
    ma = np.where(np.isnan(ma), np.nanmedian(ma), ma)
    ma = np.where(ma < 1e-6, 1e-6, ma)
    surge = cur / ma - 1.0
    return pd.Series(surge, index=names)
