"""泽平宏观V2策略 — 多因子增强 + 反转信号 + 动态权重

相比V1的三大升级：
    1. 量化因子从4个扩充到16个（5日/10日动量、量价背离、波动率收缩、
       成交额变化率、振幅衰减、隔夜收益、日内收益拆解等）
    2. 加入反转信号/均值回归因子（连续大涨后自动降低动量权重）
    3. 动态权重调整（根据市场状态自动切换宏观/量化/周期权重）

核心公式：
    泽平宏观V2评分 = 宏观维度(W_macro) + 量化维度(W_quant) + 周期维度(W_cycle)
    其中 W_macro/W_quant/W_cycle 根据市场状态动态调整：
        趋势市：量化55% + 宏观30% + 周期15%
        震荡市：宏观50% + 量化25% + 周期25%
        正常市：宏观40% + 量化40% + 周期20%（同V1）
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from real_ai_r.macro.classifier import (
    CYCLE_STAGES,
    TECH_TRACKS,
    SectorClassifier,
)
from real_ai_r.macro.red_filter import RedLineFilter

logger = logging.getLogger(__name__)


# ======================================================================
# V2 策略参数
# ======================================================================

@dataclass(frozen=True)
class ZepingV2DynamicWeights:
    """动态权重配置 — 根据市场状态自动切换。"""

    # 趋势市（市场方向明确，动量有效）
    trending_macro: float = 0.25
    trending_quant: float = 0.65
    trending_cycle: float = 0.10

    # 震荡/反转市（市场不确定，宏观更稳健）
    reverting_macro: float = 0.50
    reverting_quant: float = 0.25
    reverting_cycle: float = 0.25

    # 正常市（默认，同V1）
    normal_macro: float = 0.40
    normal_quant: float = 0.40
    normal_cycle: float = 0.20


@dataclass(frozen=True)
class ZepingV2Params:
    """泽平宏观V2策略可调参数。"""

    # ---- 宏观维度（同V1） ----
    tech_base_score: float = 85.0
    cycle_base_score: float = 55.0
    neutral_base_score: float = 40.0
    tech_heat_bonus_max: float = 30.0
    shovel_seller_bonus: float = 20.0

    # ---- 量化维度（V2扩充因子权重） ----
    # 动量族
    momentum_1d_weight: float = 3.5       # 1日动量
    momentum_5d_weight: float = 3.0       # 5日动量（新增）
    momentum_10d_weight: float = 1.5      # 10日动量（新增）
    momentum_accel_weight: float = 3.0    # 动量加速度（新增）

    # 活跃度族
    turnover_weight: float = 1.0          # 当日换手率
    turnover_5d_weight: float = 0.5       # 5日平均换手率（新增）
    amount_change_weight: float = 1.0     # 成交额变化率（新增）

    # 波动族
    volatility_contract_weight: float = 2.0  # 波动率收缩（新增）
    amplitude_decay_weight: float = 1.0      # 振幅衰减（新增）

    # 价格结构族
    overnight_weight: float = 2.5         # 隔夜收益（新增）
    intraday_weight: float = 1.5          # 日内收益（新增）
    vol_price_div_weight: float = 1.0     # 量价背离（新增，负向因子）

    # 广度族（同V1）
    breadth_weight: float = 8.0           # 上涨广度
    lead_strength_weight: float = 2.0     # 领涨强度

    # 趋势族（新增）
    consecutive_up_weight: float = 1.0    # 连涨天数

    # ---- 反转信号参数 ----
    reversal_trigger_days: int = 2        # 市场连涨N天后触发反转模式
    reversal_momentum_decay: float = 0.4   # 反转模式下动量因子衰减系数
    reversal_mean_revert_bonus: float = 12.0  # 反转模式下均值回归加成

    # ---- 周期维度（同V1） ----
    hot_stage_bonus: float = 20.0
    layout_stage_bonus: float = 15.0
    adjacent_stage_bonus: float = 10.0
    far_stage_penalty: float = -5.0


# "卖铲人"赛道
SHOVEL_SELLER_TRACKS = {"chip", "new_energy_vehicle"}


# ======================================================================
# 市场状态检测
# ======================================================================

@dataclass
class MarketRegime:
    """市场状态判断结果。"""

    regime: str = "normal"               # "trending" / "reverting" / "normal"
    consecutive_up_days: int = 0         # 市场连涨天数
    consecutive_down_days: int = 0       # 市场连跌天数
    recent_volatility: float = 0.0       # 近期波动率
    avg_market_change: float = 0.0       # 近期平均涨跌
    reversal_active: bool = False        # 是否触发反转模式
    description: str = ""


# ======================================================================
# 评分结果（扩展V1）
# ======================================================================

@dataclass
class ZepingV2BoardScore:
    """单个板块的V2策略评分结果。"""

    board_name: str
    board_code: str = ""

    # 总分
    total_score: float = 0.0

    # 三维度子分
    macro_score: float = 0.0
    quant_score: float = 0.0
    cycle_score: float = 0.0

    # 宏观标签
    macro_category: str = "neutral"
    macro_sub: str = ""
    macro_display: str = ""

    # 扩展量化指标
    momentum_1d: float = 0.0
    momentum_5d: float = 0.0
    momentum_10d: float = 0.0
    momentum_accel: float = 0.0
    turnover_rate: float = 0.0
    turnover_5d: float = 0.0
    amount_change_rate: float = 0.0
    volatility_contraction: float = 0.0
    amplitude_decay: float = 0.0
    overnight_return: float = 0.0
    intraday_return: float = 0.0
    vol_price_divergence: float = 0.0
    rise_ratio: float = 0.0
    lead_stock_pct: float = 0.0
    consecutive_up_days: int = 0

    # 周期信息
    cycle_stage: int = 0
    current_hot_stage: int = 0

    # 动态权重
    applied_weights: str = ""            # "trending/reverting/normal"

    # 推荐理由
    reasons: list[str] = field(default_factory=list)


@dataclass
class ZepingV2PredictionResult:
    """V2策略预测结果汇总。"""

    predictions: list[ZepingV2BoardScore]
    current_hot_stage: int
    current_hot_stage_name: str
    market_style: str
    market_regime: MarketRegime
    total_boards: int = 0
    filtered_redline: int = 0
    strategy_summary: str = ""


# ======================================================================
# V2 策略核心
# ======================================================================

class ZepingMacroStrategyV2:
    """泽平宏观V2策略 — 多因子增强 + 反转信号 + 动态权重。

    使用方法：
        strategy = ZepingMacroStrategyV2()
        result = strategy.predict(board_df=snapshot_df, market_history=hist_df, top_n=10)
    """

    def __init__(
        self,
        dynamic_weights: ZepingV2DynamicWeights | None = None,
        params: ZepingV2Params | None = None,
    ) -> None:
        self.dw = dynamic_weights or ZepingV2DynamicWeights()
        self.params = params or ZepingV2Params()
        self.classifier = SectorClassifier()
        self.red_filter = RedLineFilter()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def predict(
        self,
        board_df: pd.DataFrame | None = None,
        market_history: pd.DataFrame | None = None,
        top_n: int = 10,
    ) -> ZepingV2PredictionResult:
        """运行泽平宏观V2策略。

        Parameters
        ----------
        board_df : pd.DataFrame | None
            板块截面数据（需含扩展因子列，见 compute_extended_factors）。
        market_history : pd.DataFrame | None
            市场历史数据（用于判断市场状态），需含 date, market_avg 列。
        top_n : int
            返回前 N 个推荐板块。
        """
        if board_df is None:
            board_df = self._fetch_board_data()
        if board_df.empty:
            return self._empty_result()

        # 1. 分类并过滤红线
        classified = self.classifier.classify_dataframe(board_df)
        redline_count = int((classified["macro_category"] == "redline").sum())
        safe_df = classified[classified["macro_category"] != "redline"].copy()

        if safe_df.empty:
            return self._empty_result()

        # 2. 判断市场状态 → 动态权重
        regime = self._detect_market_regime(market_history)
        w_macro, w_quant, w_cycle = self._get_dynamic_weights(regime)

        # 3. 判断当前周期阶段
        hot_stage, hot_stage_name = self._detect_hot_cycle_stage(classified)

        # 4. 计算赛道热度
        track_heat = self._compute_track_heat(classified)

        # 5. 为每个板块评分
        scores: list[ZepingV2BoardScore] = []
        for _, row in safe_df.iterrows():
            score = self._score_board(
                row, hot_stage, track_heat, regime, w_macro, w_quant, w_cycle,
            )
            scores.append(score)

        # 6. 排序
        scores.sort(key=lambda s: s.total_score, reverse=True)

        # 7. 判断市场风格
        market_style = self._judge_market_style(classified, hot_stage)

        # 8. 生成摘要
        top_scores = scores[:top_n]
        summary = self._build_summary(
            top_scores, market_style, hot_stage_name, regime,
        )

        return ZepingV2PredictionResult(
            predictions=top_scores,
            current_hot_stage=hot_stage,
            current_hot_stage_name=hot_stage_name,
            market_style=market_style,
            market_regime=regime,
            total_boards=len(safe_df),
            filtered_redline=redline_count,
            strategy_summary=summary,
        )

    # ------------------------------------------------------------------
    # 市场状态检测
    # ------------------------------------------------------------------

    def _detect_market_regime(
        self,
        market_history: pd.DataFrame | None,
    ) -> MarketRegime:
        """检测市场当前状态：趋势/震荡/正常。"""
        if market_history is None or market_history.empty:
            return MarketRegime(regime="normal", description="无市场历史数据，使用默认权重")

        changes = market_history["market_avg"].values
        if len(changes) < 2:
            return MarketRegime(regime="normal", description="历史数据不足")

        # 连涨/连跌天数
        consecutive_up = 0
        consecutive_down = 0
        for c in reversed(changes):
            if c > 0:
                consecutive_up += 1
            else:
                break
        for c in reversed(changes):
            if c < 0:
                consecutive_down += 1
            else:
                break

        # 近5日波动率
        recent = changes[-5:] if len(changes) >= 5 else changes
        recent_vol = float(np.std(recent))
        avg_change = float(np.mean(recent))

        # 判断市场状态
        p = self.params
        reversal_active = consecutive_up >= p.reversal_trigger_days

        if reversal_active:
            regime = "reverting"
            desc = f"市场连涨{consecutive_up}天，触发反转模式（降低动量、提升宏观权重）"
        elif consecutive_down >= p.reversal_trigger_days:
            regime = "reverting"
            desc = f"市场连跌{consecutive_down}天，触发反转模式"
        elif recent_vol < 1.0 and abs(avg_change) < 0.5:
            regime = "normal"
            desc = "市场低波动震荡，使用均衡权重"
        elif abs(avg_change) > 1.0:
            regime = "trending"
            desc = f"市场趋势明确（均涨{avg_change:+.2f}%），提升量化权重"
        else:
            regime = "normal"
            desc = "市场正常状态，使用均衡权重"

        return MarketRegime(
            regime=regime,
            consecutive_up_days=consecutive_up,
            consecutive_down_days=consecutive_down,
            recent_volatility=recent_vol,
            avg_market_change=avg_change,
            reversal_active=reversal_active,
            description=desc,
        )

    def _get_dynamic_weights(
        self,
        regime: MarketRegime,
    ) -> tuple[float, float, float]:
        """根据市场状态返回动态权重。"""
        dw = self.dw
        if regime.regime == "trending":
            return dw.trending_macro, dw.trending_quant, dw.trending_cycle
        if regime.regime == "reverting":
            return dw.reverting_macro, dw.reverting_quant, dw.reverting_cycle
        return dw.normal_macro, dw.normal_quant, dw.normal_cycle

    # ------------------------------------------------------------------
    # 三维度评分
    # ------------------------------------------------------------------

    def _score_board(
        self,
        row: pd.Series,
        hot_stage: int,
        track_heat: dict[str, float],
        regime: MarketRegime,
        w_macro: float,
        w_quant: float,
        w_cycle: float,
    ) -> ZepingV2BoardScore:
        """对单个板块计算V2三维度综合评分。"""
        board_name = str(row.get("name", ""))
        board_code = str(row.get("code", ""))
        category = str(row.get("macro_category", "neutral"))
        sub_cat = str(row.get("macro_sub", ""))
        display = str(row.get("macro_display", ""))

        reasons: list[str] = []

        # ---- 宏观维度（同V1） ----
        macro_score = self._compute_macro_score(
            category, sub_cat, track_heat, reasons,
        )

        # ---- 量化维度（V2多因子） ----
        quant_score = self._compute_quant_score_v2(row, regime, reasons)

        # ---- 周期维度（同V1） ----
        cycle_stage = self._get_cycle_stage_num(category, sub_cat)
        cycle_score = self._compute_cycle_score(
            category, cycle_stage, hot_stage, reasons,
        )

        # ---- 动态加权综合 ----
        total = (
            w_macro * macro_score
            + w_quant * quant_score
            + w_cycle * cycle_score
        )

        weight_label = regime.regime

        return ZepingV2BoardScore(
            board_name=board_name,
            board_code=board_code,
            total_score=round(total, 2),
            macro_score=round(macro_score, 2),
            quant_score=round(quant_score, 2),
            cycle_score=round(cycle_score, 2),
            macro_category=category,
            macro_sub=sub_cat,
            macro_display=display,
            momentum_1d=_safe_float(row, "change_pct"),
            momentum_5d=_safe_float(row, "momentum_5d"),
            momentum_10d=_safe_float(row, "momentum_10d"),
            momentum_accel=_safe_float(row, "momentum_accel"),
            turnover_rate=_safe_float(row, "turnover_rate"),
            turnover_5d=_safe_float(row, "turnover_5d"),
            amount_change_rate=_safe_float(row, "amount_change_rate"),
            volatility_contraction=_safe_float(row, "volatility_contraction"),
            amplitude_decay=_safe_float(row, "amplitude_decay"),
            overnight_return=_safe_float(row, "overnight_return"),
            intraday_return=_safe_float(row, "intraday_return"),
            vol_price_divergence=_safe_float(row, "vol_price_divergence"),
            rise_ratio=self._calc_rise_ratio(row),
            lead_stock_pct=_safe_float(row, "lead_stock_pct"),
            consecutive_up_days=int(_safe_float(row, "consecutive_up_days")),
            cycle_stage=cycle_stage,
            current_hot_stage=hot_stage,
            applied_weights=weight_label,
            reasons=reasons,
        )

    def _compute_quant_score_v2(
        self,
        row: pd.Series,
        regime: MarketRegime,
        reasons: list[str],
    ) -> float:
        """计算V2量化维度得分 (0-100)。

        16个因子分为5族：动量族、活跃度族、波动族、价格结构族、广度族。
        反转模式下，动量族权重衰减，均值回归因子加成。
        """
        p = self.params

        # ---- 提取所有因子 ----
        momentum_1d = _safe_float(row, "change_pct")
        momentum_5d = _safe_float(row, "momentum_5d")
        momentum_10d = _safe_float(row, "momentum_10d")
        momentum_accel = _safe_float(row, "momentum_accel")

        turnover = _safe_float(row, "turnover_rate")
        turnover_5d = _safe_float(row, "turnover_5d")
        amount_change = _safe_float(row, "amount_change_rate")

        vol_contract = _safe_float(row, "volatility_contraction")
        amp_decay = _safe_float(row, "amplitude_decay")

        overnight = _safe_float(row, "overnight_return")
        intraday = _safe_float(row, "intraday_return")
        vol_price_div = _safe_float(row, "vol_price_divergence")

        rise_ratio = self._calc_rise_ratio(row)
        lead_pct = _safe_float(row, "lead_stock_pct")
        consec_up = _safe_float(row, "consecutive_up_days")

        # ---- 基础分 50 ----
        score = 50.0

        # ---- 反转模式调整系数 ----
        if regime.reversal_active:
            mom_decay = p.reversal_momentum_decay  # 0.4 = 动量贡献降60%
            revert_bonus = p.reversal_mean_revert_bonus
            reasons.append(f"⚡反转模式(连涨{regime.consecutive_up_days}天)")
        else:
            mom_decay = 1.0
            revert_bonus = 0.0

        # ---- 动量族（反转模式下衰减） ----
        mom_contrib = 0.0
        mom_contrib += momentum_1d * p.momentum_1d_weight
        mom_contrib += momentum_5d * p.momentum_5d_weight
        mom_contrib += momentum_10d * p.momentum_10d_weight
        mom_contrib += momentum_accel * p.momentum_accel_weight
        score += mom_contrib * mom_decay

        # ---- 反转信号：动量负的板块在反转模式下获得加成 ----
        if regime.reversal_active and momentum_1d < 0:
            # 昨天跌的板块，在市场过热后反而可能补涨
            revert_score = min(revert_bonus, abs(momentum_1d) * revert_bonus)
            score += revert_score
            if revert_score > 2:
                reasons.append(f"均值回归+{revert_score:.1f}")

        # ---- 活跃度族 ----
        score += turnover * p.turnover_weight
        score += turnover_5d * p.turnover_5d_weight
        # 成交额放大 = 资金关注
        if amount_change > 1.0:
            score += (amount_change - 1.0) * p.amount_change_weight * 10
            if amount_change > 1.5:
                reasons.append(f"成交额放大{amount_change:.1f}x")

        # ---- 波动族 ----
        # 波动率收缩 → 可能即将突破
        if vol_contract < 0.7 and vol_contract > 0:
            score += (0.7 - vol_contract) * p.volatility_contract_weight * 20
            reasons.append("波动率收缩(蓄势)")
        # 振幅衰减 → 趋势可能结束
        if amp_decay < -0.3:
            score += amp_decay * p.amplitude_decay_weight * 5

        # ---- 价格结构族 ----
        # 隔夜收益 > 0 表示市场看好
        score += overnight * p.overnight_weight * 5
        # 日内收益 > 0 表示盘中有承接
        score += intraday * p.intraday_weight * 5
        # 量价背离（负向因子）：涨价缩量 → 趋势不健康
        if vol_price_div < -0.3:
            score += vol_price_div * p.vol_price_div_weight * 5
            reasons.append("量价背离⚠️")

        # ---- 广度族 ----
        score += (rise_ratio - 0.5) * p.breadth_weight * 2
        score += lead_pct * p.lead_strength_weight

        # ---- 趋势持续族 ----
        if consec_up >= 3:
            if not regime.reversal_active:
                score += consec_up * p.consecutive_up_weight
                reasons.append(f"连涨{int(consec_up)}天")
            else:
                # 反转模式下连涨板块减分
                score -= consec_up * p.consecutive_up_weight * 0.5

        # ---- 动量因子详细标注 ----
        if abs(momentum_1d) > 0.5:
            reasons.append(f"1D动量{momentum_1d:+.2f}%")
        if abs(momentum_5d) > 2:
            reasons.append(f"5D动量{momentum_5d:+.1f}%")
        if turnover > 5:
            reasons.append(f"换手率{turnover:.1f}%")

        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------
    # 宏观维度（同V1）
    # ------------------------------------------------------------------

    def _compute_macro_score(
        self,
        category: str,
        sub_cat: str,
        track_heat: dict[str, float],
        reasons: list[str],
    ) -> float:
        """计算宏观维度得分 (0-100)。"""
        p = self.params

        if category == "tech":
            base = p.tech_base_score
            heat = track_heat.get(sub_cat, 50.0)
            heat_bonus = (heat / 100.0) * p.tech_heat_bonus_max
            shovel_bonus = p.shovel_seller_bonus if sub_cat in SHOVEL_SELLER_TRACKS else 0.0
            score = min(100.0, base + heat_bonus + shovel_bonus)
            reasons.append(f"科技主线({sub_cat})+{heat_bonus:.0f}热度")
            if shovel_bonus > 0:
                reasons.append("卖铲人赛道加成")
            return score

        if category == "cycle":
            reasons.append(f"周期主线({sub_cat})")
            return p.cycle_base_score

        reasons.append("其他板块")
        return p.neutral_base_score

    # ------------------------------------------------------------------
    # 周期维度（同V1）
    # ------------------------------------------------------------------

    def _compute_cycle_score(
        self,
        category: str,
        board_stage: int,
        hot_stage: int,
        reasons: list[str],
    ) -> float:
        """计算周期维度得分 (0-100)。"""
        p = self.params

        if category != "cycle" or board_stage == 0:
            return 50.0

        distance = abs(board_stage - hot_stage)

        if distance == 0:
            bonus = p.hot_stage_bonus
            reasons.append(f"顺周期(阶段{board_stage}=当前热点)")
        elif distance == 1:
            bonus = p.adjacent_stage_bonus
            reasons.append(f"近周期(阶段{board_stage})")
        else:
            bonus = p.far_stage_penalty
            reasons.append(f"远周期(阶段{board_stage})")

        if board_stage == 4:
            bonus += p.layout_stage_bonus
            reasons.append("重点布局区(农业后周期)")

        score = 50.0 + bonus
        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------
    # 辅助方法（复用V1逻辑）
    # ------------------------------------------------------------------

    def _detect_hot_cycle_stage(
        self,
        classified_df: pd.DataFrame,
    ) -> tuple[int, str]:
        """检测当前最热的周期阶段。"""
        cycle_df = classified_df[classified_df["macro_category"] == "cycle"]
        if cycle_df.empty:
            return 3, "传统能源"

        best_stage = 3
        best_change = -999.0
        best_name = "传统能源"

        for key, info in CYCLE_STAGES.items():
            keywords = info["keywords"]
            matched = cycle_df[
                cycle_df["name"].apply(
                    lambda n: any(kw in str(n) for kw in keywords)
                )
            ]
            if matched.empty:
                continue
            avg_change = matched["change_pct"].mean()
            if avg_change > best_change:
                best_change = avg_change
                best_stage = info["stage"]
                best_name = info["display"]

        return best_stage, best_name

    def _compute_track_heat(
        self,
        classified_df: pd.DataFrame,
    ) -> dict[str, float]:
        """计算各科技赛道的热度分数 (0-100)。"""
        tech_df = classified_df[classified_df["macro_category"] == "tech"]
        if tech_df.empty:
            return {}

        heat_map: dict[str, float] = {}
        for key, track in TECH_TRACKS.items():
            keywords = track["keywords"]
            matched = tech_df[
                tech_df["name"].apply(
                    lambda n: any(kw in str(n) for kw in keywords)
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
            momentum_score = max(0.0, min(100.0, 50.0 + avg_change * 10.0))
            activity_score = min(avg_turnover * 10, 100.0)
            heat = 0.60 * momentum_score + 0.40 * activity_score
            heat_map[key] = round(heat, 1)

        return heat_map

    def _judge_market_style(
        self,
        classified_df: pd.DataFrame,
        hot_stage: int,
    ) -> str:
        """判断当前市场风格。"""
        tech_df = classified_df[classified_df["macro_category"] == "tech"]
        cycle_df = classified_df[classified_df["macro_category"] == "cycle"]

        tech_avg = tech_df["change_pct"].mean() if not tech_df.empty else 0.0
        cycle_avg = cycle_df["change_pct"].mean() if not cycle_df.empty else 0.0

        diff = tech_avg - cycle_avg

        if diff > 1.0:
            return "强科技风格 — 科技主线领涨，适合重仓科技矛"
        if diff > 0.3:
            return "偏科技风格 — 科技略强于周期，均衡配置偏进攻"
        if diff > -0.3:
            return "均衡风格 — 科技与周期持平，攻防兼顾"
        if diff > -1.0:
            return "偏周期风格 — 周期略强于科技，适当增配防御"
        return "强周期风格 — 周期主线领涨，增配周期盾"

    @staticmethod
    def _calc_rise_ratio(row: pd.Series) -> float:
        """计算上涨家数占比。"""
        rise = float(row.get("rise_count", 0) or 0)
        fall = float(row.get("fall_count", 0) or 0)
        total = rise + fall
        return rise / total if total > 0 else 0.5

    @staticmethod
    def _get_cycle_stage_num(category: str, sub_cat: str) -> int:
        """获取周期阶段编号。"""
        if category != "cycle":
            return 0
        for key, info in CYCLE_STAGES.items():
            if key == sub_cat:
                return info["stage"]
        return 0

    def _build_summary(
        self,
        top_scores: list[ZepingV2BoardScore],
        market_style: str,
        hot_stage_name: str,
        regime: MarketRegime,
    ) -> str:
        """生成V2策略摘要。"""
        if not top_scores:
            return "无可推荐板块"

        tech_count = sum(1 for s in top_scores if s.macro_category == "tech")
        cycle_count = sum(1 for s in top_scores if s.macro_category == "cycle")
        neutral_count = sum(1 for s in top_scores if s.macro_category == "neutral")
        top3_names = ", ".join(s.board_name for s in top_scores[:3])

        return (
            f"[V2] 市场状态: {regime.description}\n"
            f"市场风格: {market_style}\n"
            f"当前周期热点: {hot_stage_name}\n"
            f"动态权重: {regime.regime}\n"
            f"推荐构成: 科技{tech_count}个 + 周期{cycle_count}个 + 其他{neutral_count}个\n"
            f"Top3: {top3_names}"
        )

    @staticmethod
    def _fetch_board_data() -> pd.DataFrame:
        """获取实时板块行情数据。"""
        from real_ai_r.sector.monitor import SectorMonitor
        try:
            return SectorMonitor.get_board_list("industry")
        except Exception:
            logger.warning("行业板块获取失败，尝试概念板块")
            try:
                return SectorMonitor.get_board_list("concept")
            except Exception:
                logger.error("板块数据获取失败")
                return pd.DataFrame()

    def _empty_result(self) -> ZepingV2PredictionResult:
        """返回空结果。"""
        return ZepingV2PredictionResult(
            predictions=[],
            current_hot_stage=0,
            current_hot_stage_name="未知",
            market_style="数据不可用",
            market_regime=MarketRegime(),
            total_boards=0,
            filtered_redline=0,
            strategy_summary="数据获取失败，无法生成预测",
        )

    # ------------------------------------------------------------------
    # 回测支持
    # ------------------------------------------------------------------

    def score_snapshot(
        self,
        board_df: pd.DataFrame,
        market_history: pd.DataFrame | None = None,
        top_n: int = 10,
    ) -> list[ZepingV2BoardScore]:
        """对历史板块截面数据评分（用于回测）。"""
        result = self.predict(
            board_df=board_df, market_history=market_history, top_n=top_n,
        )
        return result.predictions

    def to_dataframe(
        self,
        scores: list[ZepingV2BoardScore],
    ) -> pd.DataFrame:
        """将评分结果转换为 DataFrame。"""
        rows = []
        for s in scores:
            rows.append({
                "板块": s.board_name,
                "代码": s.board_code,
                "总分": s.total_score,
                "宏观分": s.macro_score,
                "量化分": s.quant_score,
                "周期分": s.cycle_score,
                "分类": s.macro_display or s.macro_category,
                "1D动量%": s.momentum_1d,
                "5D动量%": s.momentum_5d,
                "10D动量%": s.momentum_10d,
                "换手率%": s.turnover_rate,
                "成交额变化": s.amount_change_rate,
                "权重模式": s.applied_weights,
                "理由": " | ".join(s.reasons),
            })
        return pd.DataFrame(rows)


# ======================================================================
# 扩展因子计算工具
# ======================================================================

def compute_extended_factors(
    board_name: str,
    history_df: pd.DataFrame,
    target_date: pd.Timestamp,
) -> dict[str, float]:
    """从历史数据计算V2扩展因子。

    Parameters
    ----------
    board_name : str
        板块名称。
    history_df : pd.DataFrame
        该板块的历史日线数据（需含 date, open, high, low, close,
        volume, amount, change_pct, amplitude 列）。
    target_date : pd.Timestamp
        计算截止日期（用该日及之前的数据）。

    Returns
    -------
    dict[str, float]
        扩展因子字典，可直接合并到 snapshot DataFrame 行。
    """
    hist = history_df[history_df["date"] <= target_date].copy()
    if hist.empty:
        return _default_factors()

    n = len(hist)
    changes = hist["change_pct"].values
    volumes = hist["volume"].values
    closes = hist["close"].values
    opens = hist["open"].values if "open" in hist.columns else closes
    highs = hist["high"].values if "high" in hist.columns else closes
    lows = hist["low"].values if "low" in hist.columns else closes
    amplitudes = hist["amplitude"].values if "amplitude" in hist.columns else np.zeros(n)

    curr_change = float(changes[-1]) if n > 0 and not np.isnan(changes[-1]) else 0.0

    # ---- 动量族 ----
    momentum_5d = float(np.nansum(changes[-5:])) if n >= 5 else float(np.nansum(changes))
    momentum_10d = float(np.nansum(changes[-10:])) if n >= 10 else float(np.nansum(changes))

    avg_5d_daily = momentum_5d / min(n, 5)
    momentum_accel = curr_change - avg_5d_daily  # 加速度 = 今日 - 近期均值

    # ---- 活跃度族 ----
    if n >= 5:
        turnover_proxy = float(volumes[-1]) / float(np.mean(volumes[-5:])) * 2.0 if np.mean(volumes[-5:]) > 0 else 0.0
        turnover_5d = turnover_proxy  # 简化：用volume ratio代理
    else:
        turnover_proxy = 0.0
        turnover_5d = 0.0

    # 成交额变化率
    if n >= 6:
        avg_amount_5d = float(np.mean(volumes[-6:-1])) if np.mean(volumes[-6:-1]) > 0 else 1.0
        amount_change_rate = float(volumes[-1]) / avg_amount_5d if avg_amount_5d > 0 else 1.0
    else:
        amount_change_rate = 1.0

    # ---- 波动族 ----
    if n >= 5:
        recent_amps = amplitudes[-5:]
        avg_amp_5d = float(np.nanmean(recent_amps)) if len(recent_amps) > 0 else 1.0
        prev_5d_amps = amplitudes[-10:-5] if n >= 10 else amplitudes[:-5] if n > 5 else recent_amps
        avg_amp_prev = float(np.nanmean(prev_5d_amps)) if len(prev_5d_amps) > 0 else avg_amp_5d

        # 波动率收缩 = 近期振幅 / 前期振幅
        volatility_contraction = avg_amp_5d / avg_amp_prev if avg_amp_prev > 0 else 1.0

        # 振幅衰减趋势（线性回归斜率）
        if len(recent_amps) >= 3:
            x = np.arange(len(recent_amps))
            valid_mask = ~np.isnan(recent_amps)
            if valid_mask.sum() >= 2:
                slope = np.polyfit(x[valid_mask], recent_amps[valid_mask], 1)[0]
                amplitude_decay = float(slope)
            else:
                amplitude_decay = 0.0
        else:
            amplitude_decay = 0.0
    else:
        volatility_contraction = 1.0
        amplitude_decay = 0.0

    # ---- 价格结构族 ----
    if n >= 2 and not np.isnan(closes[-1]) and not np.isnan(opens[-1]):
        prev_close = float(closes[-2])
        curr_open = float(opens[-1])
        curr_close = float(closes[-1])

        # 隔夜收益 = 今开/昨收 - 1
        overnight_return = (curr_open / prev_close - 1) * 100 if prev_close > 0 else 0.0
        # 日内收益 = 今收/今开 - 1
        intraday_return = (curr_close / curr_open - 1) * 100 if curr_open > 0 else 0.0
    else:
        overnight_return = 0.0
        intraday_return = 0.0

    # 量价背离 = 价格变化方向 vs 量变化方向
    if n >= 2:
        price_dir = 1.0 if curr_change > 0 else -1.0
        vol_dir = 1.0 if volumes[-1] > volumes[-2] else -1.0
        # 同向 = 正常, 背离 = 量价不一致
        vol_price_divergence = price_dir * vol_dir  # +1=同向, -1=背离
        if curr_change > 0 and volumes[-1] < volumes[-2]:
            vol_price_divergence = -abs(curr_change) * 0.5  # 涨价缩量, 负向
    else:
        vol_price_divergence = 0.0

    # ---- 连涨天数 ----
    consecutive_up = 0
    for c in reversed(changes):
        if not np.isnan(c) and c > 0:
            consecutive_up += 1
        else:
            break

    return {
        "momentum_5d": momentum_5d,
        "momentum_10d": momentum_10d,
        "momentum_accel": momentum_accel,
        "turnover_5d": turnover_5d,
        "amount_change_rate": amount_change_rate,
        "volatility_contraction": volatility_contraction,
        "amplitude_decay": amplitude_decay,
        "overnight_return": overnight_return,
        "intraday_return": intraday_return,
        "vol_price_divergence": vol_price_divergence,
        "consecutive_up_days": float(consecutive_up),
    }


def _default_factors() -> dict[str, float]:
    """返回默认因子值。"""
    return {
        "momentum_5d": 0.0,
        "momentum_10d": 0.0,
        "momentum_accel": 0.0,
        "turnover_5d": 0.0,
        "amount_change_rate": 1.0,
        "volatility_contraction": 1.0,
        "amplitude_decay": 0.0,
        "overnight_return": 0.0,
        "intraday_return": 0.0,
        "vol_price_divergence": 0.0,
        "consecutive_up_days": 0.0,
    }


def _safe_float(row: pd.Series, col: str) -> float:
    """安全提取浮点数。"""
    val = row.get(col, 0)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    return float(val)
