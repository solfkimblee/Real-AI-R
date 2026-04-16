"""泽平宏观 V9.2 — Hedge 元集成策略

核心思想（基于 V9 回测失败的教训）：
  V9 理论上更先进，但 149 天数据不足以让 IC/HMM/QP 稳定。
  V5/V7/V8 有领域先验（科技白名单、红线、动态配比），在小样本上更鲁棒。
  V9 在某些窗口（W3、W6）确实赢 V8，说明存在"V9 擅长 / V8 擅长"的互补性。

V9.2 = Hedge(V5, V7, V8, V9) 在线元学习集成
  - 每日并行运行 4 个成员，记录各自"持仓收益"
  - Hedge 算法按实现收益动态分配成员权重（指数加权）
  - 最终持仓 = 按成员权重聚合各自 Top-N 板块
  - Hedge 理论保证：长期无遗憾接近事后最优；烂成员自动降权到 floor

与 V8 的接口完全兼容（drop-in 替换）:
  predict(board_df, fund_df, top_n, tech_history, cycle_history)
  record_excess(...)
  record_board_performance(...)

新增方法:
  observe_realized_returns(realized: dict[board_name, pct])
      回测/实盘每日收盘后调用，更新 Hedge 权重
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from real_ai_r.macro.zeping_strategy import (
    ZepingBoardScore,
    ZepingMacroStrategy,
    ZepingParams,
    ZepingPredictionResult,
    ZepingWeights,
)
from real_ai_r.macro.zeping_strategy_v5 import ZepingMacroStrategyV5
from real_ai_r.macro.zeping_strategy_v7 import ZepingMacroStrategyV7
from real_ai_r.macro.zeping_strategy_v8 import ZepingMacroStrategyV8
from real_ai_r.v9.combiner.hedge_ensemble import HedgeEnsemble
from real_ai_r.v9.engine import V9Config, V9Engine

logger = logging.getLogger(__name__)


# ======================================================================
# V9.2 参数
# ======================================================================

@dataclass(frozen=True)
class V92Params:
    """V9.2 元集成参数。"""

    # ---- Hedge 算法 ----
    # eta: 学习率。eta=1.0 下，连续跑赢 10 天 (0.5% 平均日超额) → log_w += 5
    # → exp(5)≈148 倍优势，足够压制落后者但不会瞬间锁死
    hedge_eta: float = 1.0
    hedge_warmup: int = 5                # 前 5 天用等权
    hedge_floor: float = 0.05             # 成员权重下限（防止永久归零，保留探索）

    # ---- 成员选择 ----
    # 可在构造时传入自定义子集，默认全部 4 个
    default_members: tuple[str, ...] = ("V5", "V7", "V8", "V9")

    # ---- V9 子引擎配置（小样本友好） ----
    # 针对回测中观察到的稀疏/短窗问题做保守化：关 HMM、关 Graph、关 QP
    # 让 V9 子成员退化为 "IC 加权 + Top-K 等权"（V9-Lite 模式）
    v9_enable_regime: bool = False
    v9_enable_graph: bool = False
    v9_ic_min_samples: int = 5
    v9_max_positions: int = 10


# ======================================================================
# 辅助工具
# ======================================================================


def _selection_to_weights(
    board_names: list[str], k: int | None = None,
) -> dict[str, float]:
    """把 Top-K 名单转为等权持仓字典。"""
    if not board_names:
        return {}
    if k is not None:
        board_names = board_names[:k]
    if not board_names:
        return {}
    w = 1.0 / len(board_names)
    return {b: w for b in board_names}


def _member_return(
    weights_dict: dict[str, float], realized_returns: dict[str, float],
) -> float:
    """按权重聚合实现收益（板块 change_pct）。单位与 realized_returns 一致（% 或 小数）。"""
    if not weights_dict:
        return 0.0
    ret = 0.0
    for b, w in weights_dict.items():
        ret += w * float(realized_returns.get(b, 0.0))
    return ret


# ======================================================================
# V9.2 主类
# ======================================================================


class MetaEnsembleStrategyV92(ZepingMacroStrategy):
    """V9.2 — Hedge 元集成策略。

    与 V8 接口完全兼容，可作为 drop-in 替换。
    """

    VERSION = "V9.2"

    def __init__(
        self,
        weights: ZepingWeights | None = None,
        params: ZepingParams | None = None,
        v92_params: V92Params | None = None,
        members: list[str] | None = None,
        v9_config: V9Config | None = None,
    ) -> None:
        super().__init__(weights, params)
        self.v92 = v92_params or V92Params()
        chosen = members or list(self.v92.default_members)
        self.member_names: list[str] = list(chosen)

        # 构造成员
        self._v5: ZepingMacroStrategyV5 | None = None
        self._v7: ZepingMacroStrategyV7 | None = None
        self._v8: ZepingMacroStrategyV8 | None = None
        self._v9: V9Engine | None = None

        if "V5" in self.member_names:
            self._v5 = ZepingMacroStrategyV5(weights, params)
        if "V7" in self.member_names:
            self._v7 = ZepingMacroStrategyV7(weights, params)
        if "V8" in self.member_names:
            self._v8 = ZepingMacroStrategyV8(weights, params)
        if "V9" in self.member_names:
            cfg = v9_config or V9Config(
                enable_regime=self.v92.v9_enable_regime,
                enable_graph=self.v92.v9_enable_graph,
                ic_min_samples=self.v92.v9_ic_min_samples,
                max_positions=self.v92.v9_max_positions,
            )
            self._v9 = V9Engine(cfg)

        # Hedge 集成器
        self.hedge = HedgeEnsemble(
            members=list(self.member_names),
            eta=self.v92.hedge_eta,
            warmup=self.v92.hedge_warmup,
            floor=self.v92.hedge_floor,
        )

        # 状态
        self._tech_history: list[float] = []
        self._cycle_history: list[float] = []
        self._last_member_weights: dict[str, dict[str, float]] = {}
        self._last_ensemble_top_n: int = 10
        # 记录每次预测的成员 top-k 选择（供 observe_realized_returns 用）
        self._last_member_selections: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # 主预测接口（drop-in V8）
    # ------------------------------------------------------------------

    def predict(
        self,
        board_df: pd.DataFrame | None = None,
        fund_df: pd.DataFrame | None = None,
        top_n: int = 10,
        tech_history: list[float] | None = None,
        cycle_history: list[float] | None = None,
    ) -> ZepingPredictionResult:
        if tech_history is not None:
            self._tech_history = list(tech_history)
        if cycle_history is not None:
            self._cycle_history = list(cycle_history)
        self._last_ensemble_top_n = int(top_n)

        if board_df is None or len(board_df) == 0:
            return self._empty_result()

        # --- 1. 并行调用每个成员 ---
        member_results: dict[str, ZepingPredictionResult | None] = {}
        member_selections: dict[str, list[str]] = {}

        if self._v5 is not None:
            try:
                r = self._v5.predict(board_df=board_df, top_n=top_n)
                member_results["V5"] = r
                member_selections["V5"] = [s.board_name for s in r.predictions]
            except Exception as e:
                logger.warning("V5 failed: %s", e)
                member_results["V5"] = None
                member_selections["V5"] = []

        if self._v7 is not None:
            try:
                r = self._v7.predict(
                    board_df=board_df,
                    fund_df=fund_df,
                    top_n=top_n,
                    tech_history=tech_history,
                    cycle_history=cycle_history,
                )
                member_results["V7"] = r
                member_selections["V7"] = [s.board_name for s in r.predictions]
            except Exception as e:
                logger.warning("V7 failed: %s", e)
                member_results["V7"] = None
                member_selections["V7"] = []

        if self._v8 is not None:
            try:
                r = self._v8.predict(
                    board_df=board_df,
                    fund_df=fund_df,
                    top_n=top_n,
                    tech_history=tech_history,
                    cycle_history=cycle_history,
                )
                member_results["V8"] = r
                member_selections["V8"] = [s.board_name for s in r.predictions]
            except Exception as e:
                logger.warning("V8 failed: %s", e)
                member_results["V8"] = None
                member_selections["V8"] = []

        if self._v9 is not None:
            try:
                r9 = self._v9.predict(board_df=board_df)
                member_results["V9"] = None  # V9 产出类型不同
                # 从 V9 weights 取权重 > 0 的板块，按权重降序
                top_v9 = r9.weights.sort_values(ascending=False)
                top_v9 = top_v9[top_v9 > 1e-6]
                member_selections["V9"] = list(top_v9.index[:top_n])
            except Exception as e:
                logger.warning("V9 failed: %s", e)
                member_selections["V9"] = []

        self._last_member_selections = member_selections

        # 转换为 V5/V7/V8 各自的等权持仓
        member_weights: dict[str, dict[str, float]] = {
            m: _selection_to_weights(member_selections.get(m, []), k=top_n)
            for m in self.member_names
        }
        self._last_member_weights = member_weights

        # --- 2. 按 Hedge 权重聚合持仓得分 ---
        hedge_w = self.hedge.weights()
        if not hedge_w:
            hedge_w = {m: 1.0 / len(self.member_names) for m in self.member_names}

        # 聚合: score[b] = sum_m hedge_w[m] * member_weights[m][b]
        agg: dict[str, float] = {}
        for m, mw in member_weights.items():
            w_m = hedge_w.get(m, 0.0)
            if w_m <= 0 or not mw:
                continue
            for b, v in mw.items():
                agg[b] = agg.get(b, 0.0) + w_m * v

        if not agg:
            return self._empty_result(member_results)

        # --- 3. 按聚合分降序取 top_n ---
        sorted_boards = sorted(agg.items(), key=lambda x: -x[1])[:top_n]
        final_names = [b for b, _ in sorted_boards]

        # --- 4. 为每个最终板块填充 ZepingBoardScore（优先用 V8→V7→V5 的分） ---
        score_maps: list[dict[str, ZepingBoardScore]] = []
        for m in ("V8", "V7", "V5"):
            r = member_results.get(m)
            if r is not None:
                score_maps.append({s.board_name: s for s in r.predictions})

        predictions: list[ZepingBoardScore] = []
        for bn in final_names:
            chosen: ZepingBoardScore | None = None
            for sm in score_maps:
                if bn in sm:
                    chosen = sm[bn]
                    break
            if chosen is None:
                chosen = ZepingBoardScore(board_name=bn)
            predictions.append(chosen)

        # --- 5. 构造 result（借用任一非空 V5/V7/V8 的元数据） ---
        meta_source: ZepingPredictionResult | None = None
        for m in ("V5", "V7", "V8"):
            r = member_results.get(m)
            if r is not None:
                meta_source = r
                break

        hedge_summary = " ".join(
            f"{m}:{hedge_w.get(m, 0):.2f}" for m in self.member_names
        )
        summary = f"V9.2[Hedge]({hedge_summary}) top={top_n} agg={len(agg)}"

        if meta_source is not None:
            return ZepingPredictionResult(
                predictions=predictions,
                current_hot_stage=meta_source.current_hot_stage,
                current_hot_stage_name=meta_source.current_hot_stage_name,
                market_style=f"V9.2 | {meta_source.market_style}",
                total_boards=meta_source.total_boards,
                filtered_redline=meta_source.filtered_redline,
                strategy_summary=summary,
            )
        return ZepingPredictionResult(
            predictions=predictions,
            current_hot_stage=0,
            current_hot_stage_name="",
            market_style="V9.2",
            total_boards=len(board_df),
            filtered_redline=0,
            strategy_summary=summary,
        )

    # ------------------------------------------------------------------
    # 反馈接口 — 必须每日回测结束后调用
    # ------------------------------------------------------------------

    def observe_realized_returns(
        self, realized_returns: dict[str, float],
    ) -> dict[str, float]:
        """
        用下一日的板块实现收益更新 Hedge 权重。

        参数:
            realized_returns: {board_name: 下一日 change_pct（% 单位）}
        返回:
            {member: realized_return(%)} — 各成员基于自己持仓的实现收益，便于诊断
        """
        member_returns: dict[str, float] = {}
        for m in self.member_names:
            mw = self._last_member_weights.get(m, {})
            member_returns[m] = _member_return(mw, realized_returns)

        # 同步给 V9 子引擎（让其自身 IC/board_return 历史持续累积）
        if self._v9 is not None:
            try:
                self._v9.update_feedback(realized_returns=realized_returns)
            except Exception as e:
                logger.warning("V9 update_feedback failed: %s", e)

        # Hedge 用 % 单位收益驱动（保持与用户既有回测框架一致）
        self.hedge.update(member_returns)
        return member_returns

    # V8 兼容: 这些回调仍转发给 V8 子成员（保留其回撤制动）
    def record_excess(self, daily_excess: float) -> None:
        if self._v8 is not None:
            self._v8.record_excess(daily_excess)

    def record_board_performance(
        self, board_name: str, excess: float,
    ) -> None:
        if self._v8 is not None:
            self._v8.record_board_performance(board_name, excess)

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------

    def get_hedge_weights(self) -> dict[str, float]:
        return self.hedge.weights()

    def get_last_member_selections(self) -> dict[str, list[str]]:
        return dict(self._last_member_selections)

    def reset(self) -> None:
        self.hedge.reset()
        self._tech_history.clear()
        self._cycle_history.clear()
        self._last_member_weights.clear()
        self._last_member_selections.clear()
        if self._v9 is not None:
            self._v9.reset()
        # V5/V7/V8 没有 reset 方法，只在 predict 时由 tech_history/cycle_history 覆盖

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def _empty_result(
        self, member_results: dict | None = None,
    ) -> ZepingPredictionResult:
        return ZepingPredictionResult(
            predictions=[],
            current_hot_stage=0,
            current_hot_stage_name="",
            market_style="V9.2|empty",
            total_boards=0,
            filtered_redline=0,
            strategy_summary="V9.2 empty (no members succeeded)",
        )


__all__ = ["MetaEnsembleStrategyV92", "V92Params"]
