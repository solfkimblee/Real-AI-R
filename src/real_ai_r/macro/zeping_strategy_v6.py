"""泽平宏观V6 — V5 + 智能板块轮换

V6 = V5原始输出 + 板块内智能轮换

设计哲学：
  V5的核心α来自科技板块超配（9.1/10），因此V6绝不减少科技比例。
  V6的改进方向是：在科技板块内部做智能轮换——
  用5日动量趋势识别正在走弱的科技板块，替换为正在走强的科技备选板块。

  同时，仅在极端市场条件下做少量跨类别替换（极端大跌/反转）。

核心原则：
  1. 科技板块数量 >= V5的数量（绝不减少）
  2. 科技内部轮换：弱势科技 → 强势科技备选
  3. 拖累板块仅在动量为负时替换为同类科技备选
  4. 极端保护仅替换1个板块且门槛极高
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
# V6 参数
# ======================================================================

@dataclass(frozen=True)
class V6OverlayParams:
    """V6后处理增强层参数。"""

    # ---- 模块1：科技内部轮换（核心模块） ----
    # 识别V5 Top10中5日动量最弱的科技板块，替换为备选中动量最强的科技板块
    intra_tech_rotation: bool = True
    # 5日动量阈值：只替换5日动量低于此值的板块
    weak_tech_threshold: float = -5.0      # 5日累计跌>5%才视为"弱势"
    # 备选板块5日动量阈值：替换板块必须高于此值
    strong_tech_threshold: float = 2.0     # 5日累计涨>2%才视为"强势"
    # 最多替换几个
    intra_tech_max_replace: int = 2

    # ---- 模块2：拖累板块智能替换 ----
    # 仅替换为同类别（科技→科技）的备选
    drag_board_replace: bool = True
    drag_require_weak_momentum: bool = True  # 5日动量<0才替换

    # ---- 模块3：极端大跌保护（默认禁用，减少科技→防止拉低表现） ----
    bear_enabled: bool = False
    bear_threshold: float = -2.5
    bear_replace_count: int = 1

    # ---- 模块4：大涨后反转保护（默认禁用） ----
    reversal_enabled: bool = False
    reversal_single_day: float = 3.0
    reversal_3d_sum: float = 6.0
    reversal_consecutive: int = 5
    reversal_replace_count: int = 1

    # ---- 模块5：板块轮动信号（默认禁用） ----
    rotation_enabled: bool = False
    rotation_lookback: int = 5
    rotation_threshold: float = -8.0


# 已知拖累板块
DRAG_BOARDS = {
    "计算机设备",   # 日均超额-0.210%
}


class ZepingMacroStrategyV6(ZepingMacroStrategy):
    """泽平宏观V6 — V5 + 智能板块轮换。"""

    VERSION = "V6"

    def __init__(
        self,
        weights: ZepingWeights | None = None,
        params: ZepingParams | None = None,
        v6_params: V6OverlayParams | None = None,
    ) -> None:
        super().__init__(weights, params)
        self._v5 = ZepingMacroStrategyV5(weights, params)
        self.classifier = SectorClassifier(tech_tracks=TECH_TRACKS_V5)
        self._shovel_seller_tracks = SHOVEL_SELLER_TRACKS_V5
        self.v6 = v6_params or V6OverlayParams()
        self._market_history: list[float] = []
        self._tech_history: list[float] = []
        self._cycle_history: list[float] = []

    def predict(
        self,
        board_df: pd.DataFrame | None = None,
        fund_df: pd.DataFrame | None = None,
        top_n: int = 10,
        market_history: list[float] | None = None,
        tech_history: list[float] | None = None,
        cycle_history: list[float] | None = None,
    ) -> ZepingPredictionResult:
        """V6预测 — 先跑V5，再做智能轮换。"""
        if market_history is not None:
            self._market_history = list(market_history)
        if tech_history is not None:
            self._tech_history = list(tech_history)
        if cycle_history is not None:
            self._cycle_history = list(cycle_history)

        # Step 1: V5完整预测（含更多备选）
        v5_result = self._v5.predict(board_df=board_df, top_n=top_n + 15)
        if not v5_result.predictions:
            return v5_result

        all_v5_scores = v5_result.predictions
        top_n_scores = all_v5_scores[:top_n]
        backup_scores = all_v5_scores[top_n:]

        # 获取5日动量映射
        mom_5d_map: dict[str, float] = {}
        if board_df is not None and "momentum_5d" in board_df.columns:
            for _, row in board_df.iterrows():
                name = str(row.get("name", ""))
                mom_5d_map[name] = float(row.get("momentum_5d", 0) or 0)

        # 当前市场状态
        market_mean = 0.0
        if board_df is not None and not board_df.empty and "change_pct" in board_df.columns:
            market_mean = board_df["change_pct"].mean()

        # Step 2: 依次应用后处理模块
        modified = list(top_n_scores)
        reasons_log: list[str] = []

        # 模块1: 科技内部轮换（核心改进）
        if self.v6.intra_tech_rotation:
            modified, backup_scores, log = self._apply_intra_tech_rotation(
                modified, backup_scores, mom_5d_map,
            )
            reasons_log.extend(log)

        # 模块2: 拖累板块替换（仅替换为同类科技备选）
        if self.v6.drag_board_replace:
            modified, backup_scores, log = self._replace_drag_boards(
                modified, backup_scores, mom_5d_map,
            )
            reasons_log.extend(log)

        # 模块3: 极端大跌保护
        if self.v6.bear_enabled and market_mean < self.v6.bear_threshold:
            modified, backup_scores, log = self._apply_bear_protection(
                modified, backup_scores,
            )
            reasons_log.extend(log)

        # 模块4: 反转保护
        if self.v6.reversal_enabled and self._check_reversal_trigger():
            modified, backup_scores, log = self._apply_reversal_protection(
                modified, backup_scores,
            )
            reasons_log.extend(log)

        # 模块5: 轮动信号
        rotation = self._check_rotation_signal() if self.v6.rotation_enabled else "neutral"
        if rotation == "cycle_strong":
            modified, backup_scores, log = self._apply_rotation_swap(
                modified, backup_scores,
            )
            reasons_log.extend(log)

        # Step 3: 生成V6结果
        market_style = v5_result.market_style
        if reasons_log:
            market_style += f" [V6: {', '.join(reasons_log)}]"

        summary = v5_result.strategy_summary
        if reasons_log:
            summary += f"\n[V6后处理: {'; '.join(reasons_log)}]"

        return ZepingPredictionResult(
            predictions=modified[:top_n],
            current_hot_stage=v5_result.current_hot_stage,
            current_hot_stage_name=v5_result.current_hot_stage_name,
            market_style=market_style,
            total_boards=v5_result.total_boards,
            filtered_redline=v5_result.filtered_redline,
            strategy_summary=summary,
        )

    # ------------------------------------------------------------------
    # 模块1：科技内部轮换（核心模块）
    # ------------------------------------------------------------------

    def _apply_intra_tech_rotation(
        self,
        top_scores: list[ZepingBoardScore],
        backup: list[ZepingBoardScore],
        mom_5d_map: dict[str, float],
    ) -> tuple[list[ZepingBoardScore], list[ZepingBoardScore], list[str]]:
        """科技内部轮换：弱势科技 → 强势科技备选。

        保持科技板块总数不变，只在科技内部做优胜劣汰。
        """
        v6 = self.v6
        logs: list[str] = []

        # 找出top中的弱势科技板块（5日动量 < weak_tech_threshold）
        weak_tech: list[tuple[int, ZepingBoardScore, float]] = []
        for i, s in enumerate(top_scores):
            if s.macro_category == "tech":
                mom = mom_5d_map.get(s.board_name, 0.0)
                if mom < v6.weak_tech_threshold:
                    weak_tech.append((i, s, mom))

        if not weak_tech:
            return top_scores, backup, logs

        # 按5日动量升序（最弱的优先替换）
        weak_tech.sort(key=lambda x: x[2])

        # 找备选中的强势科技板块（5日动量 > strong_tech_threshold）
        strong_tech_backup: list[tuple[ZepingBoardScore, float]] = []
        for s in backup:
            if s.macro_category == "tech":
                mom = mom_5d_map.get(s.board_name, 0.0)
                if mom > v6.strong_tech_threshold:
                    strong_tech_backup.append((s, mom))

        if not strong_tech_backup:
            return top_scores, backup, logs

        # 按5日动量降序（最强的优先补入）
        strong_tech_backup.sort(key=lambda x: x[1], reverse=True)

        replaced = 0
        result = list(top_scores)
        remaining_backup = list(backup)

        for idx, old_score, old_mom in weak_tech:
            if replaced >= v6.intra_tech_max_replace:
                break
            if not strong_tech_backup:
                break

            new_score, new_mom = strong_tech_backup.pop(0)

            # 只在动量差距足够大时替换（避免无意义的微调）
            if new_mom - old_mom < 3.0:
                continue

            result[idx] = new_score
            remaining_backup = [
                s for s in remaining_backup
                if s.board_name != new_score.board_name
            ]
            replaced += 1
            logs.append(
                f"科技轮换:{old_score.board_name}({old_mom:+.1f}%)"
                f"→{new_score.board_name}({new_mom:+.1f}%)"
            )

        return result, remaining_backup, logs

    # ------------------------------------------------------------------
    # 模块2：拖累板块替换（优先替换为科技备选）
    # ------------------------------------------------------------------

    def _replace_drag_boards(
        self,
        top_scores: list[ZepingBoardScore],
        backup: list[ZepingBoardScore],
        mom_5d_map: dict[str, float],
    ) -> tuple[list[ZepingBoardScore], list[ZepingBoardScore], list[str]]:
        """替换拖累板块为科技备选（保持科技比例不变）。"""
        logs: list[str] = []
        result = list(top_scores)
        remaining_backup = list(backup)

        for i, score in enumerate(result):
            if score.board_name not in DRAG_BOARDS:
                continue
            if not remaining_backup:
                continue

            # 5日动量为正时不替换
            if self.v6.drag_require_weak_momentum:
                mom = mom_5d_map.get(score.board_name, 0.0)
                if mom >= 0:
                    continue

            # 优先找科技备选（保持科技比例）
            tech_backup = [s for s in remaining_backup if s.macro_category == "tech"]
            if tech_backup:
                replacement = tech_backup[0]
            else:
                replacement = remaining_backup[0]

            old_name = result[i].board_name
            result[i] = replacement
            remaining_backup = [
                s for s in remaining_backup
                if s.board_name != replacement.board_name
            ]
            logs.append(f"质量过滤:{old_name}→{replacement.board_name}")

        return result, remaining_backup, logs

    # ------------------------------------------------------------------
    # 模块3：极端大跌保护
    # ------------------------------------------------------------------

    def _apply_bear_protection(
        self,
        top_scores: list[ZepingBoardScore],
        backup: list[ZepingBoardScore],
    ) -> tuple[list[ZepingBoardScore], list[ZepingBoardScore], list[str]]:
        """极端大跌：替换1个高动量科技为周期/防御板块。"""
        v6 = self.v6
        logs: list[str] = []

        tech_in_top = [
            (i, s) for i, s in enumerate(top_scores)
            if s.macro_category == "tech"
        ]
        tech_in_top.sort(key=lambda x: x[1].momentum_1d, reverse=True)

        if len(tech_in_top) <= 7:
            return top_scores, backup, logs

        non_tech_backup = [
            s for s in backup
            if s.macro_category in ("cycle", "neutral")
        ]

        replaced = 0
        result = list(top_scores)
        remaining_backup = list(backup)

        for idx, score in tech_in_top:
            if replaced >= v6.bear_replace_count:
                break
            if not non_tech_backup:
                break

            replacement = non_tech_backup.pop(0)
            old_name = result[idx].board_name
            result[idx] = replacement
            remaining_backup = [
                s for s in remaining_backup
                if s.board_name != replacement.board_name
            ]
            replaced += 1
            logs.append(f"大跌保护:{old_name}→{replacement.board_name}")

        return result, remaining_backup, logs

    # ------------------------------------------------------------------
    # 模块4：大涨后反转保护
    # ------------------------------------------------------------------

    def _check_reversal_trigger(self) -> bool:
        history = self._market_history
        if not history:
            return False

        v6 = self.v6
        if history[-1] > v6.reversal_single_day:
            return True
        if len(history) >= v6.reversal_consecutive:
            if all(d > 0 for d in history[-v6.reversal_consecutive:]):
                return True
        if len(history) >= 3:
            if sum(history[-3:]) > v6.reversal_3d_sum:
                return True
        return False

    def _apply_reversal_protection(
        self,
        top_scores: list[ZepingBoardScore],
        backup: list[ZepingBoardScore],
    ) -> tuple[list[ZepingBoardScore], list[ZepingBoardScore], list[str]]:
        """反转保护：替换动量最高的1个板块。"""
        v6 = self.v6
        logs: list[str] = []

        indexed = [(i, s) for i, s in enumerate(top_scores)]
        indexed.sort(key=lambda x: x[1].momentum_1d, reverse=True)

        low_mom_backup = sorted(backup, key=lambda s: s.momentum_1d)

        replaced = 0
        result = list(top_scores)
        remaining_backup = list(backup)

        for idx, score in indexed:
            if replaced >= v6.reversal_replace_count:
                break
            if not low_mom_backup:
                break
            if score.momentum_1d <= 1.5:
                break

            replacement = low_mom_backup.pop(0)
            old_name = result[idx].board_name
            result[idx] = replacement
            remaining_backup = [
                s for s in remaining_backup
                if s.board_name != replacement.board_name
            ]
            replaced += 1
            logs.append(f"反转保护:{old_name}→{replacement.board_name}")

        return result, remaining_backup, logs

    # ------------------------------------------------------------------
    # 模块5：板块轮动信号
    # ------------------------------------------------------------------

    def _check_rotation_signal(self) -> str:
        tech_hist = self._tech_history
        cycle_hist = self._cycle_history
        v6 = self.v6

        if len(tech_hist) < v6.rotation_lookback or \
           len(cycle_hist) < v6.rotation_lookback:
            return "neutral"

        n = v6.rotation_lookback
        diff = sum(
            t - c for t, c in zip(tech_hist[-n:], cycle_hist[-n:])
        )

        if diff < v6.rotation_threshold:
            return "cycle_strong"
        elif diff > abs(v6.rotation_threshold):
            return "tech_strong"
        return "neutral"

    def _apply_rotation_swap(
        self,
        top_scores: list[ZepingBoardScore],
        backup: list[ZepingBoardScore],
    ) -> tuple[list[ZepingBoardScore], list[ZepingBoardScore], list[str]]:
        """轮动：科技持续跑输周期时，替换1个最弱科技为最强周期。"""
        logs: list[str] = []

        tech_in_top = [
            (i, s) for i, s in enumerate(top_scores)
            if s.macro_category == "tech"
        ]
        if not tech_in_top:
            return top_scores, backup, logs

        tech_in_top.sort(key=lambda x: x[1].total_score)
        weakest_idx, weakest = tech_in_top[0]

        cycle_backup = [s for s in backup if s.macro_category == "cycle"]
        if not cycle_backup:
            return top_scores, backup, logs

        best_cycle = cycle_backup[0]
        result = list(top_scores)
        result[weakest_idx] = best_cycle
        remaining = [s for s in backup if s.board_name != best_cycle.board_name]
        logs.append(f"轮动:{weakest.board_name}→{best_cycle.board_name}")

        return result, remaining, logs

    # ------------------------------------------------------------------
    # 覆盖父类方法以使用V5赛道映射
    # ------------------------------------------------------------------

    def _compute_track_heat(self, classified_df: pd.DataFrame) -> dict[str, float]:
        return self._v5._compute_track_heat(classified_df)

    def _compute_macro_score(
        self, category: str, sub_cat: str,
        track_heat: dict[str, float], reasons: list[str],
    ) -> float:
        return self._v5._compute_macro_score(category, sub_cat, track_heat, reasons)
