"""资金流因子 — 依赖 akshare `stock_sector_fund_flow_rank` 数据。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_ai_r.v9.factors.base import FactorContext, registry


def _col(board_df: pd.DataFrame, col: str) -> pd.Series:
    if col not in board_df.columns:
        return pd.Series(
            np.zeros(len(board_df)), index=board_df["name"].values,
        )
    v = pd.to_numeric(board_df[col], errors="coerce").fillna(0.0)
    return pd.Series(v.values, index=board_df["name"].values)


@registry.register("main_inflow", direction=1, group="fund_flow")
def main_inflow(board_df: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """主力净流入额（亿元）。"""
    # akshare 字段: 主力净额 / main_net_inflow / 今日主力净流入-净额
    for col in ("main_net_inflow", "主力净额", "今日主力净流入-净额"):
        if col in board_df.columns:
            return _col(board_df, col)
    return pd.Series(
        np.zeros(len(board_df)), index=board_df["name"].values,
    )


@registry.register("main_inflow_pct", direction=1, group="fund_flow")
def main_inflow_pct(board_df: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """主力净流入占比。"""
    for col in ("main_net_inflow_pct", "主力净占比", "今日主力净流入-净占比"):
        if col in board_df.columns:
            return _col(board_df, col)
    return pd.Series(
        np.zeros(len(board_df)), index=board_df["name"].values,
    )


@registry.register("super_large_inflow", direction=1, group="fund_flow")
def super_large_inflow(
    board_df: pd.DataFrame, ctx: FactorContext,
) -> pd.Series:
    """超大单净流入。"""
    for col in ("super_large_net", "超大单净额", "今日超大单净流入-净额"):
        if col in board_df.columns:
            return _col(board_df, col)
    return pd.Series(
        np.zeros(len(board_df)), index=board_df["name"].values,
    )
