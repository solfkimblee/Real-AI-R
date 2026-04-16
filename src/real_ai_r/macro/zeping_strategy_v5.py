"""泽平宏观V5 — 修复科技赛道映射关系

V5 = V1 + 完整的科技赛道关键词映射

核心变更：
1. 科技赛道从6个扩展到9个（新增：清洁能源、通信、消费电子）
2. 关键词覆盖率从3%（15/496板块）提升至~22%（~108/496板块）
3. 修复AI医疗赛道0匹配问题（关键词与EM板块名不匹配）
4. 修复"租赁"误匹配（移除"算力租赁"关键词）
5. 卖铲人赛道扩展（新增清洁能源、通信）

策略框架不变：泽平宏观评分 = 宏观维度(40%) + 量化维度(40%) + 周期维度(20%)
量化因子不变：1日动量×5 + 换手率×2 + 上涨广度×15 + 领涨强度×3
"""

from __future__ import annotations

import logging

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
    ZepingWeights,
)

logger = logging.getLogger(__name__)


class ZepingMacroStrategyV5(ZepingMacroStrategy):
    """泽平宏观V5 — 修复科技赛道映射，覆盖率从3%提升至~22%。

    与V1完全相同的评分框架，仅更换分类器的关键词映射。
    这使得V1 vs V5的对比是纯粹的"映射修复"效果评估。
    """

    VERSION = "V5"

    def __init__(
        self,
        weights: ZepingWeights | None = None,
        params: ZepingParams | None = None,
    ) -> None:
        super().__init__(weights, params)
        # 使用V5扩展的科技赛道映射
        self.classifier = SectorClassifier(tech_tracks=TECH_TRACKS_V5)
        # V5卖铲人赛道
        self._shovel_seller_tracks = SHOVEL_SELLER_TRACKS_V5

    # ------------------------------------------------------------------
    # 覆盖父类中硬编码 TECH_TRACKS / SHOVEL_SELLER_TRACKS 的方法
    # ------------------------------------------------------------------

    def _compute_macro_score(
        self,
        category: str,
        sub_cat: str,
        track_heat: dict[str, float],
        reasons: list[str],
    ) -> float:
        """计算宏观维度得分 (0-100)。使用V5卖铲人赛道。"""
        p = self.params

        if category == "tech":
            base = p.tech_base_score
            # 赛道热度加成
            heat = track_heat.get(sub_cat, 50.0)
            heat_bonus = (heat / 100.0) * p.tech_heat_bonus_max
            # V5卖铲人加成
            shovel_bonus = (
                p.shovel_seller_bonus
                if sub_cat in self._shovel_seller_tracks
                else 0.0
            )
            score = min(100.0, base + heat_bonus + shovel_bonus)
            reasons.append(f"科技主线({sub_cat})+{heat_bonus:.0f}热度")
            if shovel_bonus > 0:
                reasons.append("卖铲人赛道加成")
            return score

        if category == "cycle":
            base = p.cycle_base_score
            reasons.append(f"周期主线({sub_cat})")
            return base

        # neutral
        reasons.append("其他板块")
        return p.neutral_base_score

    def _compute_track_heat(
        self,
        classified_df: pd.DataFrame,
    ) -> dict[str, float]:
        """计算各科技赛道的热度分数 (0-100)。使用V5赛道映射。"""
        tech_df = classified_df[classified_df["macro_category"] == "tech"]
        if tech_df.empty:
            return {}

        heat_map: dict[str, float] = {}
        # 使用V5赛道映射（而非父类硬编码的TECH_TRACKS）
        for key, track in TECH_TRACKS_V5.items():
            keywords = track["keywords"]
            matched = tech_df[
                tech_df["name"].apply(
                    lambda n, kws=keywords: any(kw in str(n) for kw in kws)
                )
            ]
            if matched.empty:
                heat_map[key] = 30.0
                continue

            avg_change = matched["change_pct"].mean()
            avg_turnover = (
                matched["turnover_rate"].mean()
                if "turnover_rate" in matched.columns
                else 0.0
            )

            # 涨幅得分 (60%) + 活跃度得分 (40%)
            momentum_score = max(0.0, min(100.0, 50.0 + avg_change * 10.0))
            activity_score = min(avg_turnover * 10, 100.0)
            heat = 0.60 * momentum_score + 0.40 * activity_score

            heat_map[key] = round(heat, 1)

        return heat_map
