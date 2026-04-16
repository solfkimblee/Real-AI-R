"""事件/催化剂因子 — 接入 catalyst 模块或 extra_signals。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_ai_r.v9.factors.base import FactorContext, registry


@registry.register("catalyst_score", direction=1, group="event")
def catalyst_score(board_df: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """从 extra_signals 读取 catalyst 模块给的事件评分。"""
    names = board_df["name"].values
    values = [
        ctx.extra_signals.get(n, {}).get("catalyst_score", 0.0) for n in names
    ]
    return pd.Series(values, index=names, dtype=float)


@registry.register("policy_tailwind", direction=1, group="event")
def policy_tailwind(board_df: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """政策利好评分（来自 catalyst）。"""
    names = board_df["name"].values
    values = [
        ctx.extra_signals.get(n, {}).get("policy_score", 0.0) for n in names
    ]
    return pd.Series(values, index=names, dtype=float)


@registry.register("earnings_revision", direction=1, group="event")
def earnings_revision(board_df: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """研报业绩上调比例。"""
    names = board_df["name"].values
    values = [
        ctx.extra_signals.get(n, {}).get("eps_revision", 0.0) for n in names
    ]
    return pd.Series(values, index=names, dtype=float)


@registry.register("redline_block", direction=-1, group="event")
def redline_block(board_df: pd.DataFrame, ctx: FactorContext) -> pd.Series:
    """红线禁区标志（-1 direction 后 = 强制扣分）。"""
    names = board_df["name"].values
    values = [
        1.0 if ctx.extra_signals.get(n, {}).get("redline", False) else 0.0
        for n in names
    ]
    return pd.Series(values, index=names, dtype=float)
