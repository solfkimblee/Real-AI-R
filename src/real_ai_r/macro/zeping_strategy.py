"""泽平宏观策略 — 深度融合泽平投资方法论 + 规则引擎

核心公式：
    泽平宏观评分 = 宏观维度(40%) + 量化维度(40%) + 周期维度(20%)

宏观维度（泽平方法论）：
    - 科技主线板块：基础加分 + 所属赛道热度加成
    - 周期主线板块：基础加分 + 周期阶段匹配加成（顺周期>逆周期）
    - 红线禁区板块：直接排除，不参与评分
    - 其他板块：无宏观加成

量化维度（规则引擎增强版）：
    - 1日动量（趋势延续效应，A股散户市场最有效因子）
    - 换手率（活跃度信号）
    - 上涨广度（板块内多数个股上涨=趋势健康）
    - 领涨强度（龙头效应）

周期维度（五段论轮动）：
    - 识别当前最热周期阶段
    - 板块所处阶段与当前热阶段距离越近得分越高
    - "重点布局区"（阶段四农业后周期）额外加分

设计原则：
    1. "宏观定方向" — 科技主线天然高权重，红线直接排除
    2. "周期定节奏" — 顺周期板块在轮动中获得加成
    3. "简单规则不过拟合" — 保留规则引擎的鲁棒性，不使用ML训练
    4. "卖铲人优先" — 算力/芯片/电力设备等基础设施赛道额外加分
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from real_ai_r.macro.classifier import (
    CYCLE_STAGES,
    TECH_TRACKS,
    SectorClassifier,
)
from real_ai_r.macro.red_filter import RedLineFilter

logger = logging.getLogger(__name__)


# ======================================================================
# 策略参数
# ======================================================================

@dataclass(frozen=True)
class ZepingWeights:
    """泽平宏观策略各维度权重。"""

    macro: float = 0.40       # 宏观维度权重
    quant: float = 0.40       # 量化维度权重
    cycle: float = 0.20       # 周期维度权重


@dataclass(frozen=True)
class ZepingParams:
    """泽平宏观策略可调参数。"""

    # 宏观维度
    tech_base_score: float = 70.0        # 科技主线基础分
    cycle_base_score: float = 55.0       # 周期主线基础分
    neutral_base_score: float = 40.0     # 其他板块基础分
    tech_heat_bonus_max: float = 30.0    # 赛道热度最大加成
    shovel_seller_bonus: float = 10.0    # "卖铲人"赛道额外加分

    # 量化维度
    momentum_weight: float = 5.0         # 动量系数（与规则引擎一致）
    turnover_weight: float = 2.0         # 换手率系数
    breadth_weight: float = 15.0         # 上涨广度系数
    lead_strength_weight: float = 3.0    # 领涨强度系数

    # 周期维度
    hot_stage_bonus: float = 20.0        # 热阶段匹配加成
    layout_stage_bonus: float = 15.0     # "重点布局区"加成（阶段四）
    adjacent_stage_bonus: float = 10.0   # 相邻阶段加成
    far_stage_penalty: float = -5.0      # 远离阶段减分


# "卖铲人"赛道 — 基础设施层，优先兑现业绩
SHOVEL_SELLER_TRACKS = {"chip", "new_energy_vehicle"}


# ======================================================================
# 评分结果
# ======================================================================

@dataclass
class ZepingBoardScore:
    """单个板块的泽平宏观策略评分结果。"""

    board_name: str
    board_code: str = ""

    # 总分
    total_score: float = 0.0

    # 三维度子分
    macro_score: float = 0.0
    quant_score: float = 0.0
    cycle_score: float = 0.0

    # 宏观标签
    macro_category: str = "neutral"      # tech/cycle/redline/neutral
    macro_sub: str = ""                  # 细分赛道/阶段
    macro_display: str = ""              # 中文显示名

    # 量化指标
    momentum_1d: float = 0.0
    turnover_rate: float = 0.0
    rise_ratio: float = 0.0
    lead_stock_pct: float = 0.0

    # 周期信息
    cycle_stage: int = 0                 # 所属周期阶段 (0=非周期)
    current_hot_stage: int = 0           # 当前最热阶段

    # 推荐理由
    reasons: list[str] = field(default_factory=list)


@dataclass
class ZepingPredictionResult:
    """泽平宏观策略预测结果汇总。"""

    predictions: list[ZepingBoardScore]
    current_hot_stage: int               # 当前最热周期阶段
    current_hot_stage_name: str          # 当前最热阶段中文名
    market_style: str                    # 市场风格判断
    total_boards: int = 0                # 参与评分板块数
    filtered_redline: int = 0            # 被过滤的红线板块数
    strategy_summary: str = ""           # 策略摘要


# ======================================================================
# 策略核心
# ======================================================================

class ZepingMacroStrategy:
    """泽平宏观策略 — 融合宏观方法论与规则引擎。

    使用方法：
        strategy = ZepingMacroStrategy()
        result = strategy.predict(top_n=10)
        for board in result.predictions:
            print(f"{board.board_name}: {board.total_score:.1f}分")
    """

    def __init__(
        self,
        weights: ZepingWeights | None = None,
        params: ZepingParams | None = None,
    ) -> None:
        self.weights = weights or ZepingWeights()
        self.params = params or ZepingParams()
        self.classifier = SectorClassifier()
        self.red_filter = RedLineFilter()

    # ------------------------------------------------------------------
    # 主入口：预测
    # ------------------------------------------------------------------

    def predict(
        self,
        board_df: pd.DataFrame | None = None,
        fund_df: pd.DataFrame | None = None,
        top_n: int = 10,
    ) -> ZepingPredictionResult:
        """运行泽平宏观策略，预测明日热门板块。

        Parameters
        ----------
        board_df : pd.DataFrame | None
            板块行情数据。如果为 None，自动从 SectorMonitor 获取。
        fund_df : pd.DataFrame | None
            资金流向数据。如果为 None，自动获取。
        top_n : int
            返回前 N 个推荐板块。

        Returns
        -------
        ZepingPredictionResult
            包含评分排名、市场风格判断、策略摘要的完整结果。
        """
        # 1. 获取数据
        if board_df is None:
            board_df = self._fetch_board_data()
        if board_df.empty:
            return self._empty_result()

        # 2. 分类并过滤红线
        classified = self.classifier.classify_dataframe(board_df)
        redline_count = (classified["macro_category"] == "redline").sum()
        safe_df = classified[classified["macro_category"] != "redline"].copy()

        if safe_df.empty:
            return self._empty_result()

        # 3. 判断当前周期阶段
        hot_stage, hot_stage_name = self._detect_hot_cycle_stage(classified)

        # 4. 计算赛道热度映射
        track_heat = self._compute_track_heat(classified)

        # 5. 为每个板块评分
        scores: list[ZepingBoardScore] = []
        for _, row in safe_df.iterrows():
            score = self._score_board(row, hot_stage, track_heat)
            scores.append(score)

        # 6. 排序
        scores.sort(key=lambda s: s.total_score, reverse=True)

        # 7. 判断市场风格
        market_style = self._judge_market_style(classified, hot_stage)

        # 8. 生成策略摘要
        top_scores = scores[:top_n]
        summary = self._build_summary(top_scores, market_style, hot_stage_name)

        return ZepingPredictionResult(
            predictions=top_scores,
            current_hot_stage=hot_stage,
            current_hot_stage_name=hot_stage_name,
            market_style=market_style,
            total_boards=len(safe_df),
            filtered_redline=redline_count,
            strategy_summary=summary,
        )

    def predict_from_snapshot(
        self,
        snapshot_df: pd.DataFrame,
        top_n: int = 10,
    ) -> ZepingPredictionResult:
        """从板块截面快照预测（用于回测）。

        与 predict() 相同逻辑，但不自动获取数据，直接使用传入的快照。
        """
        return self.predict(board_df=snapshot_df, top_n=top_n)

    # ------------------------------------------------------------------
    # 三维度评分
    # ------------------------------------------------------------------

    def _score_board(
        self,
        row: pd.Series,
        hot_stage: int,
        track_heat: dict[str, float],
    ) -> ZepingBoardScore:
        """对单个板块计算三维度综合评分。"""
        board_name = str(row.get("name", ""))
        board_code = str(row.get("code", ""))
        category = str(row.get("macro_category", "neutral"))
        sub_cat = str(row.get("macro_sub", ""))
        display = str(row.get("macro_display", ""))

        reasons: list[str] = []

        # ---- 宏观维度 ----
        macro_score = self._compute_macro_score(
            category, sub_cat, track_heat, reasons,
        )

        # ---- 量化维度 ----
        quant_score = self._compute_quant_score(row, reasons)

        # ---- 周期维度 ----
        cycle_stage = self._get_cycle_stage_num(category, sub_cat)
        cycle_score = self._compute_cycle_score(
            category, cycle_stage, hot_stage, reasons,
        )

        # ---- 综合 ----
        total = (
            self.weights.macro * macro_score
            + self.weights.quant * quant_score
            + self.weights.cycle * cycle_score
        )

        return ZepingBoardScore(
            board_name=board_name,
            board_code=board_code,
            total_score=round(total, 2),
            macro_score=round(macro_score, 2),
            quant_score=round(quant_score, 2),
            cycle_score=round(cycle_score, 2),
            macro_category=category,
            macro_sub=sub_cat,
            macro_display=display,
            momentum_1d=float(row.get("change_pct", 0) or 0),
            turnover_rate=float(row.get("turnover_rate", 0) or 0),
            rise_ratio=self._calc_rise_ratio(row),
            lead_stock_pct=float(row.get("lead_stock_pct", 0) or 0),
            cycle_stage=cycle_stage,
            current_hot_stage=hot_stage,
            reasons=reasons,
        )

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
            # 赛道热度加成
            heat = track_heat.get(sub_cat, 50.0)
            heat_bonus = (heat / 100.0) * p.tech_heat_bonus_max
            # "卖铲人"加成
            shovel_bonus = p.shovel_seller_bonus if sub_cat in SHOVEL_SELLER_TRACKS else 0.0
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

    def _compute_quant_score(
        self,
        row: pd.Series,
        reasons: list[str],
    ) -> float:
        """计算量化维度得分 (0-100)。

        基于规则引擎的核心逻辑，增加广度和领涨强度。
        """
        p = self.params

        momentum = float(row.get("change_pct", 0) or 0)
        turnover = float(row.get("turnover_rate", 0) or 0)
        rise_ratio = self._calc_rise_ratio(row)
        lead_pct = float(row.get("lead_stock_pct", 0) or 0)

        # 基础分 50 + 各因子贡献
        score = 50.0
        score += momentum * p.momentum_weight
        score += turnover * p.turnover_weight
        score += (rise_ratio - 0.5) * p.breadth_weight * 2  # 0.5=中性，>0.5加分
        score += lead_pct * p.lead_strength_weight

        score = max(0.0, min(100.0, score))

        if abs(momentum) > 0.5:
            reasons.append(f"动量{momentum:+.2f}%")
        if turnover > 5:
            reasons.append(f"换手率{turnover:.1f}%")

        return score

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
            return 50.0  # 非周期板块得中性分

        distance = abs(board_stage - hot_stage)

        if distance == 0:
            # 正好在最热阶段
            bonus = p.hot_stage_bonus
            reasons.append(f"顺周期(阶段{board_stage}=当前热点)")
        elif distance == 1:
            # 相邻阶段
            bonus = p.adjacent_stage_bonus
            reasons.append(f"近周期(阶段{board_stage})")
        else:
            # 远离阶段
            bonus = p.far_stage_penalty
            reasons.append(f"远周期(阶段{board_stage})")

        # 阶段四"重点布局区"特殊加成
        if board_stage == 4:
            bonus += p.layout_stage_bonus
            reasons.append("重点布局区(农业后周期)")

        score = 50.0 + bonus
        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _detect_hot_cycle_stage(
        self,
        classified_df: pd.DataFrame,
    ) -> tuple[int, str]:
        """检测当前最热的周期阶段。

        通过各周期阶段板块的平均涨幅判断。
        """
        cycle_df = classified_df[classified_df["macro_category"] == "cycle"]
        if cycle_df.empty:
            return 3, "传统能源"  # 默认阶段三

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

            # 涨幅得分 (60%) + 活跃度得分 (40%)
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
        top_scores: list[ZepingBoardScore],
        market_style: str,
        hot_stage_name: str,
    ) -> str:
        """生成策略摘要文字。"""
        if not top_scores:
            return "无可推荐板块"

        tech_count = sum(1 for s in top_scores if s.macro_category == "tech")
        cycle_count = sum(1 for s in top_scores if s.macro_category == "cycle")
        neutral_count = sum(1 for s in top_scores if s.macro_category == "neutral")

        top3_names = ", ".join(s.board_name for s in top_scores[:3])

        return (
            f"市场风格: {market_style}\n"
            f"当前周期热点: {hot_stage_name}\n"
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

    def _empty_result(self) -> ZepingPredictionResult:
        """返回空结果。"""
        return ZepingPredictionResult(
            predictions=[],
            current_hot_stage=0,
            current_hot_stage_name="未知",
            market_style="数据不可用",
            total_boards=0,
            filtered_redline=0,
            strategy_summary="数据获取失败，无法生成预测",
        )

    # ------------------------------------------------------------------
    # 回测支持：从历史快照评分
    # ------------------------------------------------------------------

    def score_snapshot(
        self,
        board_df: pd.DataFrame,
        top_n: int = 10,
    ) -> list[ZepingBoardScore]:
        """对历史板块截面数据评分（用于回测）。

        Parameters
        ----------
        board_df : pd.DataFrame
            板块截面数据，需含列：name, change_pct, turnover_rate,
            rise_count, fall_count, lead_stock_pct
        top_n : int
            返回前 N 个。

        Returns
        -------
        list[ZepingBoardScore]
            按评分排序的板块列表。
        """
        result = self.predict(board_df=board_df, top_n=top_n)
        return result.predictions

    def to_dataframe(
        self,
        scores: list[ZepingBoardScore],
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
                "涨跌幅%": s.momentum_1d,
                "换手率%": s.turnover_rate,
                "理由": " | ".join(s.reasons),
            })
        return pd.DataFrame(rows)
