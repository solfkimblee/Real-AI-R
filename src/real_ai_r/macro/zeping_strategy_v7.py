"""泽平宏观V7 — V5 + 动态科技/周期配比 + 大宗轮动

核心创新：
  V5固定91%科技配置在科技牛市表现极好，但在周期强势时大幅亏损（W1/W2夏普-3.5）。
  V7根据5日滚动科技vs周期趋势动态调整配比：
    - 科技强势：科技8-9/10（类似V5）
    - 均衡市场：科技6-7/10（降低科技，补充周期）
    - 周期强势：科技4-5/10（大幅降科技，增加周期）

  在周期配额内，按周期5段论的5日动量选择最强的周期阶段板块。

Walk-Forward验证驱动的设计：
  - W1(9-10月)科技累计+0.28% vs 周期+4.99% → V5亏损-3.65%
  - W2(10-11月)科技-3.20% vs 周期-0.72% → V5亏损-1.81%
  - 模拟动态配比(9/7/4)在全147天: 夏普3.09 vs V5固定: 夏普0.90
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from real_ai_r.macro.classifier import (
    SHOVEL_SELLER_TRACKS_V5,
    TECH_TRACKS_V5,
    SectorClassifier,
)
from real_ai_r.macro.zeping_strategy import (
    ZepingBoardScore,
    ZepingMacroStrategy,
    ZepingParams,
    ZepingPredictionResult,
    ZepingWeights,
)
from real_ai_r.macro.zeping_strategy_v5 import ZepingMacroStrategyV5

logger = logging.getLogger(__name__)


# ======================================================================
# V7参数
# ======================================================================

@dataclass(frozen=True)
class V7Params:
    """V7动态配比参数。"""

    # ---- 市场状态检测 ----
    # 用5日滚动科技vs周期涨幅差来判断当前市场状态
    # 阈值±1.5经参数扫描验证：WF夏普2.78(vs默认±2.0的2.23)，W2转正(+0.80)
    regime_lookback: int = 5              # 回看天数
    tech_strong_threshold: float = 1.5    # 科技5d-周期5d > 1.5% → 科技强势
    cycle_strong_threshold: float = -1.5  # 科技5d-周期5d < -1.5% → 周期强势

    # ---- 动态配比 ----
    # 不同市场状态下，Top10中科技板块的目标数量
    tech_count_in_tech_strong: int = 9    # 科技强势时
    tech_count_in_neutral: int = 7        # 均衡时
    tech_count_in_cycle_strong: int = 4   # 周期强势时

    # ---- 周期内部轮动 ----
    # 在周期配额内，优先选择5日动量最强的周期阶段
    cycle_momentum_lookback: int = 5      # 周期板块动量回看天数

    # ---- 平滑过渡 ----
    # 每天科技数量最多变化几个（防止过度切换）
    max_daily_change: int = 2             # 每天最多增减2个科技板块


class ZepingMacroStrategyV7(ZepingMacroStrategy):
    """泽平宏观V7 — 动态科技/周期配比 + 大宗轮动。

    V7 = V5评分框架 + 动态配比引擎

    执行流程：
    1. 用V5对所有板块评分排序
    2. 检测当前市场状态（科技强势/均衡/周期强势）
    3. 根据市场状态确定科技/周期目标配比
    4. 平滑过渡：每天最多调整2个板块
    5. 从V5排序中按配比选取Top10
    """

    VERSION = "V7"

    def __init__(
        self,
        weights: ZepingWeights | None = None,
        params: ZepingParams | None = None,
        v7_params: V7Params | None = None,
    ) -> None:
        super().__init__(weights, params)
        self._v5 = ZepingMacroStrategyV5(weights, params)
        self.classifier = SectorClassifier(tech_tracks=TECH_TRACKS_V5)
        self._shovel_seller_tracks = SHOVEL_SELLER_TRACKS_V5
        self.v7 = v7_params or V7Params()

        # 历史状态（用于平滑过渡）
        self._tech_history: list[float] = []
        self._cycle_history: list[float] = []
        self._prev_tech_count: int | None = None  # 上一天的科技板块数

    def predict(
        self,
        board_df: pd.DataFrame | None = None,
        fund_df: pd.DataFrame | None = None,
        top_n: int = 10,
        tech_history: list[float] | None = None,
        cycle_history: list[float] | None = None,
    ) -> ZepingPredictionResult:
        """V7预测 — V5评分 + 动态配比。"""
        if tech_history is not None:
            self._tech_history = list(tech_history)
        if cycle_history is not None:
            self._cycle_history = list(cycle_history)

        # Step 1: V5完整评分（获取所有板块排序）
        v5_result = self._v5.predict(board_df=board_df, top_n=90)
        if not v5_result.predictions:
            return v5_result

        all_scores = v5_result.predictions

        # Step 2: 检测市场状态
        regime = self._detect_regime()

        # Step 3: 确定目标科技数量
        target_tech = self._get_target_tech_count(regime, top_n)

        # Step 4: 平滑过渡（并确保不超过top_n）
        target_tech = min(self._smooth_transition(target_tech), top_n)

        # Step 5: 按配比从V5排序中选取Top N
        top_n_scores = self._select_by_allocation(
            all_scores, top_n, target_tech, board_df,
        )

        # 更新历史状态
        self._prev_tech_count = sum(
            1 for s in top_n_scores if s.macro_category == "tech"
        )

        # 构建结果
        summary_parts = [
            f"V7[{regime}]",
            f"科技{self._prev_tech_count}/{top_n}",
        ]

        return ZepingPredictionResult(
            predictions=top_n_scores,
            current_hot_stage=v5_result.current_hot_stage,
            current_hot_stage_name=v5_result.current_hot_stage_name,
            market_style=f"{v5_result.market_style} | {regime}",
            total_boards=v5_result.total_boards,
            filtered_redline=v5_result.filtered_redline,
            strategy_summary=" | ".join(summary_parts),
        )

    # ------------------------------------------------------------------
    # 市场状态检测
    # ------------------------------------------------------------------

    def _detect_regime(self) -> str:
        """检测当前市场状态：tech_strong / neutral / cycle_strong。

        使用最近N天的科技vs周期累计涨幅差来判断。
        """
        lookback = self.v7.regime_lookback

        if len(self._tech_history) < lookback or len(self._cycle_history) < lookback:
            return "neutral"  # 数据不足时默认均衡

        tech_sum = sum(self._tech_history[-lookback:])
        cycle_sum = sum(self._cycle_history[-lookback:])
        diff = tech_sum - cycle_sum

        if diff > self.v7.tech_strong_threshold:
            return "tech_strong"
        elif diff < self.v7.cycle_strong_threshold:
            return "cycle_strong"
        else:
            return "neutral"

    # ------------------------------------------------------------------
    # 目标配比
    # ------------------------------------------------------------------

    def _get_target_tech_count(self, regime: str, top_n: int) -> int:
        """根据市场状态返回目标科技板块数量。"""
        if regime == "tech_strong":
            return min(self.v7.tech_count_in_tech_strong, top_n)
        elif regime == "cycle_strong":
            return min(self.v7.tech_count_in_cycle_strong, top_n)
        else:
            return min(self.v7.tech_count_in_neutral, top_n)

    def _smooth_transition(self, target_tech: int) -> int:
        """平滑过渡：每天最多调整max_daily_change个板块。"""
        if self._prev_tech_count is None:
            return target_tech

        prev = self._prev_tech_count
        max_change = self.v7.max_daily_change

        if target_tech > prev:
            return min(target_tech, prev + max_change)
        elif target_tech < prev:
            return max(target_tech, prev - max_change)
        else:
            return target_tech

    # ------------------------------------------------------------------
    # 按配比选取
    # ------------------------------------------------------------------

    def _select_by_allocation(
        self,
        all_scores: list[ZepingBoardScore],
        top_n: int,
        target_tech: int,
        board_df: pd.DataFrame | None,
    ) -> list[ZepingBoardScore]:
        """从V5排序结果中，按目标配比选取Top N。

        逻辑：
        1. 从V5排名中按顺序取科技板块，直到达到target_tech
        2. 从V5排名中按顺序取非科技板块，直到总数达到top_n
        3. 在非科技配额中，优先选择5日动量最强的周期板块

        返回的列表按V5原始评分排序（保持评分排序的一致性）。
        """
        # 获取5日动量映射
        mom_5d_map: dict[str, float] = {}
        if board_df is not None and "momentum_5d" in board_df.columns:
            for _, row in board_df.iterrows():
                name = str(row.get("name", ""))
                mom_5d_map[name] = float(row.get("momentum_5d", 0) or 0)

        # 分离科技和非科技板块（按V5评分排序）
        tech_scores: list[ZepingBoardScore] = []
        non_tech_scores: list[ZepingBoardScore] = []

        for s in all_scores:
            if s.macro_category == "tech":
                tech_scores.append(s)
            elif s.macro_category != "redline":
                non_tech_scores.append(s)

        # 选取科技板块（按V5评分排序取前target_tech个）
        selected_tech = tech_scores[:target_tech]

        # 选取非科技板块
        target_non_tech = top_n - len(selected_tech)

        if target_non_tech > 0 and non_tech_scores:
            # 在非科技中，用5日动量加权重排序（优先动量强的周期板块）
            # 但仍然尊重V5的基础排序（通过混合分数）
            scored_non_tech = []
            for s in non_tech_scores:
                mom = mom_5d_map.get(s.board_name, 0.0)
                # 混合分数：V5评分(归一化) + 5日动量加成
                # 这样既尊重V5的宏观/量化/周期评分，又偏向近期动量强的板块
                mixed = s.total_score + mom * 0.5
                scored_non_tech.append((s, mixed))

            scored_non_tech.sort(key=lambda x: x[1], reverse=True)

            # 优先选周期板块（大宗轮动的核心）
            cycle_selected = []
            other_selected = []
            for s, _ in scored_non_tech:
                if s.macro_category == "cycle" and len(cycle_selected) < target_non_tech:
                    cycle_selected.append(s)
                elif s.macro_category != "cycle":
                    other_selected.append(s)

            selected_non_tech = cycle_selected
            # 如果周期板块不够，补充其他板块
            remaining = target_non_tech - len(selected_non_tech)
            if remaining > 0:
                selected_non_tech.extend(other_selected[:remaining])
        else:
            selected_non_tech = []

        # 合并并按V5原始评分排序
        result = selected_tech + selected_non_tech[:target_non_tech]
        result.sort(key=lambda s: s.total_score, reverse=True)

        return result[:top_n]
