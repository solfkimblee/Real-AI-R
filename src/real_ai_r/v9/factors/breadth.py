"""广度类因子 — 反映板块内部的健康度。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_ai_r.v9.factors.base import FactorContext, registry


@registry.register("rise_ratio", direction=1, group="breadth")
def rise_ratio(board_df: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """上涨家数 / (上涨+下跌)。"""
    rise = pd.to_numeric(
        board_df.get("rise_count", 0), errors="coerce",
    ).fillna(0.0)
    fall = pd.to_numeric(
        board_df.get("fall_count", 0), errors="coerce",
    ).fillna(0.0)
    denom = (rise + fall).replace(0, np.nan)
    ratio = (rise / denom).fillna(0.5)
    return pd.Series(ratio.values, index=board_df["name"].values)


@registry.register("lead_strength", direction=1, group="breadth")
def lead_strength(board_df: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """领涨股的涨幅 — 代表板块龙头强度。"""
    if "lead_stock_pct" not in board_df.columns:
        return pd.Series(
            np.zeros(len(board_df)), index=board_df["name"].values,
        )
    v = pd.to_numeric(board_df["lead_stock_pct"], errors="coerce").fillna(0.0)
    return pd.Series(v.values, index=board_df["name"].values)


@registry.register("dispersion", direction=-1, group="breadth")
def dispersion(board_df: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """板块内部分化度 — 领涨股 - 板块均涨 越大越分化，分化越大越差。

    direction=-1: 分化越小（即领涨-板块均涨差距小=板块整体向上）越好。
    """
    lead = pd.to_numeric(
        board_df.get("lead_stock_pct", 0), errors="coerce",
    ).fillna(0.0)
    mean = pd.to_numeric(
        board_df.get("change_pct", 0), errors="coerce",
    ).fillna(0.0)
    disp = (lead - mean).abs()
    return pd.Series(disp.values, index=board_df["name"].values)
