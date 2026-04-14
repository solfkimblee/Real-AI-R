"""周期轮动仪表盘 — 大宗商品五段论追踪

基于泽平宏观的"五段论"演进模型，追踪大宗商品周期轮动：
阶段一：贵金属（金、银）→ 阶段二：基本金属（铜、铝）→
阶段三：传统能源（煤、油）→ 阶段四：农业后周期（化肥、种子）→
阶段五：必选消费（食品、日化）

通过实时板块数据计算各阶段的"温度"指标，判断当前所处位置。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from real_ai_r.macro.classifier import CYCLE_STAGES
from real_ai_r.sector.monitor import SectorMonitor

logger = logging.getLogger(__name__)


@dataclass
class StageStatus:
    """单个周期阶段的状态快照。"""

    stage: int                # 阶段编号 1-5
    key: str                  # 阶段 key
    display: str              # 中文名
    icon: str                 # 图标
    description: str          # 阶段说明
    framework_status: str     # 框架定位（如"高位区"、"重点布局区"）
    temperature: float        # 综合温度 0-100
    avg_change_pct: float     # 当日平均涨跌幅
    avg_turnover: float       # 平均换手率
    fund_flow_score: float    # 资金流向得分 0-100
    momentum_score: float     # 动量得分 0-100
    matched_boards: list[str] # 匹配到的板块名称


class CycleTracker:
    """大宗商品五段论周期追踪器。

    实时计算各阶段板块的综合"温度"指标，用于判断当前
    处于五段论的哪个阶段，以及各阶段的热度变化趋势。
    """

    # 温度计算权重
    TEMP_WEIGHTS = {
        "momentum": 0.35,    # 涨幅动量
        "fund_flow": 0.30,   # 资金流向
        "activity": 0.20,    # 活跃度（换手率）
        "breadth": 0.15,     # 上涨广度
    }

    def __init__(self, cycle_stages: dict | None = None) -> None:
        self.stages = cycle_stages or CYCLE_STAGES

    def track(self) -> list[StageStatus]:
        """追踪所有五个周期阶段的当前状态。

        Returns
        -------
        list[StageStatus]
            5个阶段的状态快照，按阶段编号排序。
        """
        logger.info("开始追踪周期轮动状态...")

        # 获取行业板块数据
        try:
            board_df = SectorMonitor.get_board_list("industry")
        except Exception:
            logger.warning("行业板块数据获取失败，尝试概念板块")
            try:
                board_df = SectorMonitor.get_board_list("concept")
            except Exception:
                logger.error("板块数据获取失败")
                return self._empty_stages()

        # 获取资金流向
        try:
            fund_df = SectorMonitor.get_fund_flow("今日", "行业资金流")
        except Exception:
            logger.warning("资金流向数据获取失败")
            fund_df = pd.DataFrame()

        results = []
        for key, stage_info in self.stages.items():
            status = self._compute_stage_status(
                key, stage_info, board_df, fund_df,
            )
            results.append(status)

        results.sort(key=lambda s: s.stage)
        return results

    def get_current_stage(self) -> StageStatus | None:
        """获取当前最热的周期阶段。

        Returns
        -------
        StageStatus | None
            温度最高的阶段，如果无数据则返回 None。
        """
        stages = self.track()
        if not stages:
            return None
        return max(stages, key=lambda s: s.temperature)

    def get_stage_ranking(self) -> list[StageStatus]:
        """获取各阶段按温度排名。"""
        stages = self.track()
        return sorted(stages, key=lambda s: s.temperature, reverse=True)

    def _compute_stage_status(
        self,
        key: str,
        stage_info: dict,
        board_df: pd.DataFrame,
        fund_df: pd.DataFrame,
    ) -> StageStatus:
        """计算单个阶段的状态。"""
        keywords = stage_info["keywords"]

        # 匹配板块
        matched = board_df[
            board_df["name"].apply(lambda n: any(kw in str(n) for kw in keywords))
        ]
        matched_names = matched["name"].tolist() if not matched.empty else []

        if matched.empty:
            return StageStatus(
                stage=stage_info["stage"],
                key=key,
                display=stage_info["display"],
                icon=stage_info["icon"],
                description=stage_info["desc"],
                framework_status=stage_info["status"],
                temperature=0.0,
                avg_change_pct=0.0,
                avg_turnover=0.0,
                fund_flow_score=0.0,
                momentum_score=0.0,
                matched_boards=matched_names,
            )

        # 计算涨幅动量得分
        avg_change = matched["change_pct"].mean()
        momentum_score = self._change_to_score(avg_change)

        # 计算活跃度
        avg_turnover = (
            matched["turnover_rate"].mean()
            if "turnover_rate" in matched.columns
            else 0.0
        )
        activity_score = min(avg_turnover * 10, 100)  # 换手率10%=满分

        # 计算上涨广度
        if "rise_count" in matched.columns and "fall_count" in matched.columns:
            total_rise = matched["rise_count"].sum()
            total_fall = matched["fall_count"].sum()
            total = total_rise + total_fall
            breadth_score = (total_rise / total * 100) if total > 0 else 50.0
        else:
            breadth_score = 50.0

        # 计算资金流向得分
        fund_flow_score = self._compute_fund_flow_score(
            matched_names, fund_df,
        )

        # 综合温度
        temperature = (
            self.TEMP_WEIGHTS["momentum"] * momentum_score
            + self.TEMP_WEIGHTS["fund_flow"] * fund_flow_score
            + self.TEMP_WEIGHTS["activity"] * activity_score
            + self.TEMP_WEIGHTS["breadth"] * breadth_score
        )

        return StageStatus(
            stage=stage_info["stage"],
            key=key,
            display=stage_info["display"],
            icon=stage_info["icon"],
            description=stage_info["desc"],
            framework_status=stage_info["status"],
            temperature=round(temperature, 1),
            avg_change_pct=round(avg_change, 2),
            avg_turnover=round(avg_turnover, 2),
            fund_flow_score=round(fund_flow_score, 1),
            momentum_score=round(momentum_score, 1),
            matched_boards=matched_names,
        )

    @staticmethod
    def _change_to_score(change_pct: float) -> float:
        """将涨跌幅转换为 0-100 的得分。

        涨幅 >=5% → 100分
        涨幅 0% → 50分
        跌幅 <=-5% → 0分
        """
        score = 50.0 + change_pct * 10.0
        return max(0.0, min(100.0, score))

    @staticmethod
    def _compute_fund_flow_score(
        board_names: list[str],
        fund_df: pd.DataFrame,
    ) -> float:
        """计算匹配板块的资金流向综合得分。"""
        if fund_df.empty or not board_names:
            return 50.0

        # 查找名称列
        name_col = None
        for col in ["名称", "name"]:
            if col in fund_df.columns:
                name_col = col
                break
        if name_col is None:
            return 50.0

        # 查找净流入列
        inflow_col = None
        for col in fund_df.columns:
            if "主力净流入-净额" in col or "主力净流入" in col:
                inflow_col = col
                break
        if inflow_col is None:
            return 50.0

        # 匹配板块资金流向
        matched_fund = fund_df[
            fund_df[name_col].apply(
                lambda n: any(bn in str(n) or str(n) in bn for bn in board_names)
            )
        ]

        if matched_fund.empty:
            return 50.0

        # 净流入占全市场比例
        total_inflow = fund_df[inflow_col].sum()
        stage_inflow = matched_fund[inflow_col].sum()

        if total_inflow == 0:
            return 50.0

        ratio = stage_inflow / abs(total_inflow)
        # 映射到 0-100：ratio 为正表示净流入
        score = 50.0 + ratio * 50.0
        return max(0.0, min(100.0, score))

    def _empty_stages(self) -> list[StageStatus]:
        """数据获取失败时返回空状态。"""
        results = []
        for key, info in self.stages.items():
            results.append(StageStatus(
                stage=info["stage"],
                key=key,
                display=info["display"],
                icon=info["icon"],
                description=info["desc"],
                framework_status=info["status"],
                temperature=0.0,
                avg_change_pct=0.0,
                avg_turnover=0.0,
                fund_flow_score=0.0,
                momentum_score=0.0,
                matched_boards=[],
            ))
        results.sort(key=lambda s: s.stage)
        return results
