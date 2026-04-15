"""泽平宏观V1.1核心引擎 — V1 + 独立反转保护

设计理念（借鉴幻方量化"系统化、工程化、持续迭代"）：
    策略的α来自于对数据更深层次的处理，而不仅是因子数量的堆砌。
    当历史数据受限时，对实时数据的深度挖掘和状态管理比强行构建复杂因子更重要。

架构：
    V1.1 = V1核心引擎（不修改任何因子） + 独立反转保护模块

反转保护逻辑：
    1. 检测市场连涨天数（需传入market_history）
    2. 当连涨≥3天时触发反转保护
    3. 从V1 Top10中识别动量分最高的2个板块
    4. 从全市场跌幅前20%中挑选宏观分最高的2个板块替换
    5. 正常日完全等于V1，零损耗

进化路线：
    模块A (V1.1) → 模块B (EM实时增强) → 模块C (自建库后V2复活)
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

logger = logging.getLogger(__name__)


# ======================================================================
# 反转保护参数
# ======================================================================

@dataclass(frozen=True)
class ReversalProtectionParams:
    """独立反转保护模块参数。

    多重触发条件（满足任一即触发）：
    1. 市场连涨≥trigger_days天
    2. 前一日市场涨幅≥single_day_threshold%
    3. 最近N日累计涨幅≥cumulative_threshold%
    """

    # 触发条件1: 连涨天数
    trigger_days: int = 3              # 市场连涨N天后触发
    # 触发条件2: 单日涨幅阈值
    single_day_threshold: float = 2.0  # 前一日市场均涨>2%触发
    # 触发条件3: N日累计涨幅
    cumulative_days: int = 2           # 看最近N天
    cumulative_threshold: float = 3.0  # N日累计涨幅>3%触发

    replace_count: int = 2             # 替换Top10中动量最高的N个
    oversold_percentile: float = 0.20  # 从跌幅前20%中挑选替换板块
    prefer_macro_categories: tuple[str, ...] = ("tech", "cycle")  # 替换优先宏观类别


# ======================================================================
# 反转保护状态
# ======================================================================

@dataclass
class ReversalProtectionState:
    """反转保护模块状态。"""

    triggered: bool = False
    trigger_reason: str = ""               # 触发原因
    consecutive_up_days: int = 0
    last_day_change: float = 0.0           # 前一日市场涨幅
    cumulative_change: float = 0.0         # N日累计涨幅
    replaced_out: list[str] = field(default_factory=list)   # 被替换掉的板块
    replaced_in: list[str] = field(default_factory=list)    # 替换进来的板块
    description: str = ""


# ======================================================================
# V1.1 预测结果（扩展V1）
# ======================================================================

@dataclass
class ZepingV11PredictionResult:
    """V1.1策略预测结果 = V1结果 + 反转保护状态。"""

    # V1 原始结果
    v1_result: ZepingPredictionResult

    # V1.1 最终推荐（可能经过反转替换）
    final_predictions: list[ZepingBoardScore]

    # 反转保护状态
    reversal_state: ReversalProtectionState

    # 便捷属性
    @property
    def predictions(self) -> list[ZepingBoardScore]:
        return self.final_predictions

    @property
    def current_hot_stage(self) -> int:
        return self.v1_result.current_hot_stage

    @property
    def current_hot_stage_name(self) -> str:
        return self.v1_result.current_hot_stage_name

    @property
    def market_style(self) -> str:
        return self.v1_result.market_style

    @property
    def strategy_summary(self) -> str:
        base = self.v1_result.strategy_summary
        if self.reversal_state.triggered:
            return (
                f"[V1.1 反转保护已触发] "
                f"连涨{self.reversal_state.consecutive_up_days}天, "
                f"替换: {self.reversal_state.replaced_out} → "
                f"{self.reversal_state.replaced_in}\n{base}"
            )
        return f"[V1.1 正常模式]\n{base}"


# ======================================================================
# V1.1 核心引擎
# ======================================================================

class ZepingMacroStrategyV11:
    """泽平宏观V1.1核心引擎 — V1 + 独立反转保护。

    使用方法：
        engine = ZepingMacroStrategyV11()
        result = engine.predict(board_df=snapshot, market_history=hist, top_n=10)

    正常日：完全等于V1（零损耗）
    连涨日：触发反转保护，替换动量最高的板块为超跌+宏观优质板块
    """

    def __init__(
        self,
        v1_weights: ZepingWeights | None = None,
        v1_params: ZepingParams | None = None,
        reversal_params: ReversalProtectionParams | None = None,
    ) -> None:
        self.v1_strategy = ZepingMacroStrategy(
            weights=v1_weights, params=v1_params,
        )
        self.reversal_params = reversal_params or ReversalProtectionParams()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def predict(
        self,
        board_df: pd.DataFrame | None = None,
        market_history: pd.DataFrame | None = None,
        top_n: int = 10,
    ) -> ZepingV11PredictionResult:
        """运行V1.1策略。

        Parameters
        ----------
        board_df : pd.DataFrame | None
            板块截面数据（同V1）。
        market_history : pd.DataFrame | None
            市场历史数据（用于反转保护），需含 date, market_avg 列。
        top_n : int
            返回前 N 个推荐板块。
        """
        # Step 1: 运行V1获取原始结果
        v1_result = self.v1_strategy.predict(
            board_df=board_df, top_n=top_n,
        )

        if not v1_result.predictions:
            return ZepingV11PredictionResult(
                v1_result=v1_result,
                final_predictions=[],
                reversal_state=ReversalProtectionState(
                    description="无预测结果",
                ),
            )

        # Step 2: 检测市场状态（多重触发条件）
        market_state = self._detect_market_state(market_history)
        consecutive_up = market_state["consecutive_up"]
        last_day_change = market_state["last_day_change"]
        cumulative_change = market_state["cumulative_change"]

        # Step 3: 判断是否触发反转保护（满足任一条件即触发）
        rp = self.reversal_params
        trigger_reason = ""
        should_trigger = False

        if consecutive_up >= rp.trigger_days:
            should_trigger = True
            trigger_reason = f"连涨{consecutive_up}天≥{rp.trigger_days}天"
        elif last_day_change >= rp.single_day_threshold:
            should_trigger = True
            trigger_reason = f"前日涨幅{last_day_change:+.2f}%≥{rp.single_day_threshold}%"
        elif cumulative_change >= rp.cumulative_threshold:
            should_trigger = True
            trigger_reason = (
                f"{rp.cumulative_days}日累计涨幅{cumulative_change:+.2f}%"
                f"≥{rp.cumulative_threshold}%"
            )

        if should_trigger:
            # 触发反转保护
            final_predictions, reversal_state = self._apply_reversal_protection(
                v1_result, board_df, consecutive_up, top_n,
                trigger_reason=trigger_reason,
                last_day_change=last_day_change,
                cumulative_change=cumulative_change,
            )
        else:
            # 正常模式，完全等于V1
            final_predictions = list(v1_result.predictions)
            reversal_state = ReversalProtectionState(
                triggered=False,
                consecutive_up_days=consecutive_up,
                last_day_change=last_day_change,
                cumulative_change=cumulative_change,
                description=(
                    f"正常模式（连涨{consecutive_up}天, "
                    f"前日{last_day_change:+.2f}%, "
                    f"{rp.cumulative_days}日累计{cumulative_change:+.2f}%）"
                ),
            )

        return ZepingV11PredictionResult(
            v1_result=v1_result,
            final_predictions=final_predictions,
            reversal_state=reversal_state,
        )

    # ------------------------------------------------------------------
    # 反转保护核心逻辑
    # ------------------------------------------------------------------

    def _apply_reversal_protection(
        self,
        v1_result: ZepingPredictionResult,
        board_df: pd.DataFrame | None,
        consecutive_up: int,
        top_n: int,
        trigger_reason: str = "",
        last_day_change: float = 0.0,
        cumulative_change: float = 0.0,
    ) -> tuple[list[ZepingBoardScore], ReversalProtectionState]:
        """执行反转保护：替换动量最高的板块为超跌+宏观优质板块。"""
        rp = self.reversal_params
        top_predictions = list(v1_result.predictions)

        # Step 1: 从Top10中识别动量分最高的N个板块
        sorted_by_momentum = sorted(
            top_predictions, key=lambda s: s.momentum_1d, reverse=True,
        )
        high_mom_boards = sorted_by_momentum[:rp.replace_count]
        high_mom_names = {s.board_name for s in high_mom_boards}

        # Step 2: 获取全市场评分（需要board_df）
        if board_df is not None:
            all_scores = self.v1_strategy.predict(
                board_df=board_df,
                top_n=999,  # 获取所有板块评分
            ).predictions
        else:
            all_scores = []

        # Step 3: 从全市场中找超跌板块（跌幅前20%）
        # 排除已在Top10中的板块
        top_names = {s.board_name for s in top_predictions}
        candidate_pool = [
            s for s in all_scores if s.board_name not in top_names
        ]

        # 按涨跌幅排序，取跌幅最大的前20%
        n_oversold = max(1, int(len(candidate_pool) * rp.oversold_percentile))
        oversold_boards = sorted(
            candidate_pool, key=lambda s: s.momentum_1d,
        )[:n_oversold]

        # Step 4: 从超跌板块中按宏观分最高挑选替换板块
        # 优先选科技/周期类别
        preferred = [
            s for s in oversold_boards
            if s.macro_category in rp.prefer_macro_categories
        ]
        if len(preferred) >= rp.replace_count:
            # 优先类别中按宏观分排序
            replacements = sorted(
                preferred, key=lambda s: s.macro_score, reverse=True,
            )[:rp.replace_count]
        else:
            # 不够则从全部超跌中按宏观分选
            replacements = sorted(
                oversold_boards, key=lambda s: s.macro_score, reverse=True,
            )[:rp.replace_count]

        # Step 5: 执行替换
        final_predictions = [
            s for s in top_predictions if s.board_name not in high_mom_names
        ]
        for replacement in replacements:
            # 标记为反转替换
            replacement.reasons.append(
                f"🔄反转替换(连涨{consecutive_up}天, 超跌+宏观优质)"
            )
            final_predictions.append(replacement)

        # 按总分重新排序
        final_predictions.sort(key=lambda s: s.total_score, reverse=True)
        final_predictions = final_predictions[:top_n]

        replaced_out = [s.board_name for s in high_mom_boards]
        replaced_in = [s.board_name for s in replacements]

        reversal_state = ReversalProtectionState(
            triggered=True,
            trigger_reason=trigger_reason,
            consecutive_up_days=consecutive_up,
            last_day_change=last_day_change,
            cumulative_change=cumulative_change,
            replaced_out=replaced_out,
            replaced_in=replaced_in,
            description=(
                f"反转保护触发（{trigger_reason}）| "
                f"替换: {replaced_out} → {replaced_in}"
            ),
        )

        logger.info(
            "V1.1反转保护触发: %s, 替换 %s → %s",
            trigger_reason, replaced_out, replaced_in,
        )

        return final_predictions, reversal_state

    # ------------------------------------------------------------------
    # 市场状态检测
    # ------------------------------------------------------------------

    def _detect_market_state(
        self,
        market_history: pd.DataFrame | None,
    ) -> dict:
        """检测市场状态：连涨天数、前日涨幅、N日累计涨幅。"""
        result = {
            "consecutive_up": 0,
            "last_day_change": 0.0,
            "cumulative_change": 0.0,
        }
        if market_history is None or market_history.empty:
            return result

        changes = market_history["market_avg"].values

        # 连涨天数
        consecutive_up = 0
        for c in reversed(changes):
            if not np.isnan(c) and c > 0:
                consecutive_up += 1
            else:
                break
        result["consecutive_up"] = consecutive_up

        # 前一日涨幅
        if len(changes) >= 1:
            last_val = changes[-1]
            result["last_day_change"] = float(last_val) if not np.isnan(last_val) else 0.0

        # N日累计涨幅
        rp = self.reversal_params
        n = min(rp.cumulative_days, len(changes))
        if n > 0:
            recent = changes[-n:]
            cum = sum(float(c) for c in recent if not np.isnan(c))
            result["cumulative_change"] = cum

        return result

    # ------------------------------------------------------------------
    # 回测支持
    # ------------------------------------------------------------------

    def score_snapshot(
        self,
        board_df: pd.DataFrame,
        market_history: pd.DataFrame | None = None,
        top_n: int = 10,
    ) -> list[ZepingBoardScore]:
        """对历史板块截面数据评分（用于回测）。"""
        result = self.predict(
            board_df=board_df, market_history=market_history, top_n=top_n,
        )
        return result.final_predictions
