"""泽平宏观V8 — 连续得分融合 + 双时间尺度确认 + 换手阻尼 + 回撤制动

核心创新（基于V1-V7全部经验教训自主设计）：

  V7的问题：硬性分配（9/7/4科技板块）在制度边界处有cliff效应。
  W6(02-27~03-26)被检测为cycle_strong，配4/10科技→大亏-0.96%。
  W3(11-25~12-22)neutral配7/10→略亏-0.47%。
  根因：二值化分配在阈值附近会出现误判whipsaw。

  V8解决方案：
  1. 连续制度得分 → 连续的科技/周期加减分，不做硬性数量分配
     - 制度越偏周期→给周期板块加分越多（而非强制分配周期板块数量）
     - 科技板块保留V5原始评分优势，只在极强周期信号时被替代
     - 避免V7的cliff效应：中等信号→中等调整

  2. 双时间尺度确认（3天短期 + 7天长期）
     - 短期和长期方向一致时才采信完整信号
     - 方向矛盾时信号减半→保守
     - 解决V7在W6单一5天窗口误判的问题

  3. 换手阻尼（持仓惯性）
     - 昨日Top10内的板块自动获得+2分加成
     - 自然降低换手率（减少交易成本），无需硬性持仓期限
     - 根据V5 100天分析，日均换3-5个板块→年化交易成本5-8%

  4. 滚动表现过滤
     - 最近5天日均超额为负的板块→扣分
     - 解决V5中计算机设备(-0.210%/天)等拖累板块反复入选的问题

  5. 回撤制动器
     - 最近5天累计超额 < -3% → 强制信号归零（等效于V5纯评分）
     - 避免在连续亏损时继续加码错误方向
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
# V8参数
# ======================================================================

@dataclass(frozen=True)
class V8Params:
    """V8连续融合参数。"""

    # ---- 双时间尺度制度检测 ----
    short_lookback: int = 3           # 短期回看天数
    long_lookback: int = 7            # 长期回看天数

    # ---- 连续得分融合 ----
    # regime_score > 0 时科技占优，< 0 时周期占优
    # regime_score现在基于日均涨幅差（非累计和），所以系数需要相应放大
    # 例如：日均制度差-1% → 周期板块加 1 × 25.0 = 25分
    # 参数经10组配置扫描 + sum→avg修正后重新验证，cb25_tb5最优
    cycle_bonus_per_pct: float = 25.0  # 周期板块每%日均制度偏离的加分系数
    tech_bonus_per_pct: float = 5.0    # 科技板块每%日均制度偏离的加分系数
    max_adjustment: float = 25.0       # 单板块最大调整分（防止极端值）

    # ---- 换手阻尼 ----
    holdover_bonus: float = 2.0        # 昨日持仓板块的惯性加分

    # ---- 滚动表现过滤 ----
    perf_lookback: int = 5             # 表现回看天数
    underperform_penalty: float = 3.0  # 近期表现差的板块扣分

    # ---- 回撤制动 ----
    # 阈值-10%仅在极端亏损时触发（安全网），不在正常波动中干扰策略
    # 参数扫描发现：-3%过于敏感，在W1/W2正常波动时误触发导致回退到纯V5
    drawdown_lookback: int = 5         # 回撤检测回看天数
    drawdown_threshold: float = -10.0  # 累计超额低于此值触发制动（%）

    # ---- 周期内部轮动 ----
    cycle_momentum_weight: float = 0.5  # 5日动量在非科技排序中的权重


class ZepingMacroStrategyV8(ZepingMacroStrategy):
    """泽平宏观V8 — 连续得分融合 + 双时间尺度确认 + 换手阻尼 + 回撤制动。

    V8 = V5评分框架 + 连续融合引擎

    与V7的核心区别：
    - V7: 检测制度→硬性分配(9/7/4)→从V5排名中按配额选取
    - V8: 检测制度→计算连续调整分→加到V5基础分上→统一排序选Top10
      不存在硬性配额，制度信号弱→几乎不调整→接近V5
      制度信号强→大幅调整→自然推动周期板块进入Top10

    执行流程：
    1. 用V5对所有板块评分
    2. 双时间尺度检测制度强度（连续值，非离散）
    3. 回撤制动检查
    4. 对每个板块计算制度调整分
    5. 加换手阻尼（昨日持仓加分）
    6. 加滚动表现过滤（近期弱势扣分）
    7. V5分 + 调整分 → 统一排序 → 取Top10
    """

    VERSION = "V8"

    def __init__(
        self,
        weights: ZepingWeights | None = None,
        params: ZepingParams | None = None,
        v8_params: V8Params | None = None,
    ) -> None:
        super().__init__(weights, params)
        self._v5 = ZepingMacroStrategyV5(weights, params)
        self.classifier = SectorClassifier(tech_tracks=TECH_TRACKS_V5)
        self._shovel_seller_tracks = SHOVEL_SELLER_TRACKS_V5
        self.v8 = v8_params or V8Params()

        # 状态追踪
        self._tech_history: list[float] = []
        self._cycle_history: list[float] = []
        self._prev_selections: list[str] = []      # 上一天选出的板块名
        self._excess_history: list[float] = []      # 日超额收益历史
        self._board_excess_history: dict[str, list[float]] = {}  # 各板块超额历史

    # ------------------------------------------------------------------
    # 外部接口：记录超额收益（WF验证时从外部传入）
    # ------------------------------------------------------------------

    def record_excess(self, daily_excess: float) -> None:
        """记录策略当日超额收益（用于回撤制动）。"""
        self._excess_history.append(daily_excess)

    def record_board_performance(
        self, board_name: str, excess: float,
    ) -> None:
        """记录单板块当日超额表现（用于滚动过滤）。"""
        if board_name not in self._board_excess_history:
            self._board_excess_history[board_name] = []
        self._board_excess_history[board_name].append(excess)

    # ------------------------------------------------------------------
    # 主预测入口
    # ------------------------------------------------------------------

    def predict(
        self,
        board_df: pd.DataFrame | None = None,
        fund_df: pd.DataFrame | None = None,
        top_n: int = 10,
        tech_history: list[float] | None = None,
        cycle_history: list[float] | None = None,
    ) -> ZepingPredictionResult:
        """V8预测 — V5评分 + 连续融合引擎。"""
        if tech_history is not None:
            self._tech_history = list(tech_history)
        if cycle_history is not None:
            self._cycle_history = list(cycle_history)

        # Step 1: V5完整评分（获取所有板块排序）
        v5_result = self._v5.predict(board_df=board_df, top_n=90)
        if not v5_result.predictions:
            return v5_result

        all_scores = v5_result.predictions

        # Step 2: 双时间尺度制度检测（连续值）
        regime_score = self._compute_regime_score()

        # Step 3: 回撤制动检查
        if self._is_in_drawdown():
            regime_score = 0.0  # 制动：取消所有制度调整，回退到纯V5

        # Step 4: 获取5日动量映射
        mom_5d_map: dict[str, float] = {}
        if board_df is not None and "momentum_5d" in board_df.columns:
            for _, row in board_df.iterrows():
                name = str(row.get("name", ""))
                mom_5d_map[name] = float(row.get("momentum_5d", 0) or 0)

        # Step 5: 对每个板块计算调整分并融合
        adjusted_scores: list[tuple[ZepingBoardScore, float]] = []
        for s in all_scores:
            if s.macro_category == "redline":
                continue  # 红线板块永远不选

            adj = 0.0

            # 5a. 制度调整分
            adj += self._compute_regime_adjustment(
                s, regime_score, mom_5d_map,
            )

            # 5b. 换手阻尼
            if s.board_name in self._prev_selections:
                adj += self.v8.holdover_bonus

            # 5c. 滚动表现过滤
            adj += self._compute_performance_adjustment(s.board_name)

            final_score = s.total_score + adj
            adjusted_scores.append((s, final_score))

        # Step 6: 按调整后分数排序，取Top N
        adjusted_scores.sort(key=lambda x: x[1], reverse=True)
        top_n_items = adjusted_scores[:top_n]

        # 更新持仓状态
        self._prev_selections = [s.board_name for s, _ in top_n_items]

        # 统计信息
        tech_count = sum(
            1 for s, _ in top_n_items if s.macro_category == "tech"
        )
        cycle_count = sum(
            1 for s, _ in top_n_items if s.macro_category == "cycle"
        )

        # 构建结果（保持ZepingBoardScore原始数据不变，只是排序改了）
        result_scores = [s for s, _ in top_n_items]

        regime_label = self._regime_label(regime_score)
        summary_parts = [
            f"V8[{regime_label}|rs={regime_score:+.1f}]",
            f"科技{tech_count}/周期{cycle_count}/{top_n}",
        ]
        if self._is_in_drawdown():
            summary_parts.append("制动中")

        return ZepingPredictionResult(
            predictions=result_scores,
            current_hot_stage=v5_result.current_hot_stage,
            current_hot_stage_name=v5_result.current_hot_stage_name,
            market_style=f"{v5_result.market_style} | {regime_label}",
            total_boards=v5_result.total_boards,
            filtered_redline=v5_result.filtered_redline,
            strategy_summary=" | ".join(summary_parts),
        )

    # ------------------------------------------------------------------
    # 双时间尺度制度检测
    # ------------------------------------------------------------------

    def _compute_regime_score(self) -> float:
        """计算连续制度得分。

        正值→科技占优，负值→周期占优，0→均衡/数据不足。

        使用双时间尺度确认：
        - short_diff = 近3天科技均涨 - 周期均涨
        - long_diff = 近7天科技均涨 - 周期均涨
        - 方向一致: 取保守值（绝对值较小的）
        - 方向矛盾: 信号减半（取平均的一半）
        """
        short_n = self.v8.short_lookback
        long_n = self.v8.long_lookback

        # 数据不足时返回中性
        if (
            len(self._tech_history) < long_n
            or len(self._cycle_history) < long_n
        ):
            return 0.0

        # 短期差异（用均值，消除3天vs7天的尺度差异）
        short_tech = sum(self._tech_history[-short_n:]) / short_n
        short_cycle = sum(self._cycle_history[-short_n:]) / short_n
        short_diff = short_tech - short_cycle

        # 长期差异（用均值，与短期可比）
        long_tech = sum(self._tech_history[-long_n:]) / long_n
        long_cycle = sum(self._cycle_history[-long_n:]) / long_n
        long_diff = long_tech - long_cycle

        # 双时间尺度确认
        if short_diff > 0 and long_diff > 0:
            # 双确认科技强：取保守值
            return min(short_diff, long_diff)
        elif short_diff < 0 and long_diff < 0:
            # 双确认周期强：取保守值（绝对值较小的负数）
            return max(short_diff, long_diff)
        else:
            # 方向矛盾：信号减半
            return (short_diff + long_diff) / 4.0

    # ------------------------------------------------------------------
    # 制度调整分计算
    # ------------------------------------------------------------------

    def _compute_regime_adjustment(
        self,
        score: ZepingBoardScore,
        regime_score: float,
        mom_5d_map: dict[str, float],
    ) -> float:
        """根据制度强度计算单板块的调整分。

        逻辑：
        - regime_score > 0 (科技占优): 科技板块微加分，周期板块微减分
        - regime_score < 0 (周期占优): 周期板块加分(按动量加权)，科技板块微减分
        - regime_score ≈ 0: 几乎无调整（回退到V5纯评分）

        关键设计：
        - 不对科技做大幅减分（保留V5的底分优势）
        - 周期加分与5日动量挂钩（只推动动量好的周期板块上来）
        - 用max_adjustment防止极端调整
        """
        p = self.v8
        adj = 0.0

        if score.macro_category == "tech":
            if regime_score > 0:
                # 科技强势→给科技板块小幅加分（V5已有高底分，不需要太多）
                adj = min(regime_score * p.tech_bonus_per_pct, p.max_adjustment)
            elif regime_score < 0:
                # 周期强势→科技板块小幅减分（但减分<加分，保守处理）
                # 只减一半的调整量，避免过度减持科技
                adj = max(regime_score * p.tech_bonus_per_pct * 0.5, -p.max_adjustment)

        elif score.macro_category == "cycle":
            mom_5d = mom_5d_map.get(score.board_name, 0.0)
            if regime_score < 0:
                # 周期强势→给周期板块加分（与动量正相关）
                base_bonus = abs(regime_score) * p.cycle_bonus_per_pct
                # 动量调制：只有动量为正的周期板块才获得完整加分
                # 动量为负的周期板块只获得一半加分（避免推入弱势周期板块）
                if mom_5d > 0:
                    mom_multiplier = min(1.0 + mom_5d * 0.1, 1.5)
                else:
                    mom_multiplier = 0.5
                adj = min(base_bonus * mom_multiplier, p.max_adjustment)
            elif regime_score > 0:
                # 科技强势→周期板块小幅减分
                adj = max(-regime_score * p.cycle_bonus_per_pct * 0.3, -p.max_adjustment)

        else:
            # neutral板块：不做制度调整
            pass

        return adj

    # ------------------------------------------------------------------
    # 滚动表现过滤
    # ------------------------------------------------------------------

    def _compute_performance_adjustment(self, board_name: str) -> float:
        """根据板块近期表现计算调整分。

        近期持续跑输的板块扣分，持续跑赢的不加分（V5已反映）。
        """
        history = self._board_excess_history.get(board_name, [])
        lookback = self.v8.perf_lookback

        if len(history) < lookback:
            return 0.0  # 数据不足不调整

        recent = history[-lookback:]
        avg_excess = sum(recent) / len(recent)

        if avg_excess < -0.1:
            # 近期日均跑输0.1%以上 → 扣分
            # 跑输越多扣越多，上限为underperform_penalty
            penalty = min(abs(avg_excess) * 10, self.v8.underperform_penalty)
            return -penalty

        return 0.0

    # ------------------------------------------------------------------
    # 回撤制动
    # ------------------------------------------------------------------

    def _is_in_drawdown(self) -> bool:
        """检测是否处于回撤制动状态。

        如果最近N天的累计超额收益低于阈值，触发制动。
        制动效果：regime_score强制归零，策略回退到纯V5排序。
        """
        lookback = self.v8.drawdown_lookback
        if len(self._excess_history) < lookback:
            return False

        recent_cum = sum(self._excess_history[-lookback:])
        return recent_cum < self.v8.drawdown_threshold

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _regime_label(regime_score: float) -> str:
        """将连续制度分转为可读标签。"""
        if regime_score > 2.0:
            return "强科技"
        elif regime_score > 0.5:
            return "偏科技"
        elif regime_score < -2.0:
            return "强周期"
        elif regime_score < -0.5:
            return "偏周期"
        else:
            return "均衡"
