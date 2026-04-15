"""明日热门板块预测模块

基于多因子综合评分模型预测次日可能的热门板块：
- 资金流向因子：主力净流入金额与占比
- 动量因子：当日涨幅、近期涨幅趋势
- 活跃度因子：换手率、上涨家数占比
- 领涨强度因子：板块内领涨股涨幅
"""

from __future__ import annotations

import logging

import pandas as pd

from real_ai_r.sector.monitor import SectorMonitor

logger = logging.getLogger(__name__)


class HotSectorPredictor:
    """热门板块预测器 — 多因子评分模型。"""

    # 各因子权重（可调）
    DEFAULT_WEIGHTS = {
        "fund_flow": 0.30,    # 资金流向
        "momentum": 0.25,     # 涨幅动量
        "activity": 0.20,     # 活跃度
        "breadth": 0.15,      # 上涨广度
        "lead_strength": 0.10,  # 领涨强度
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        board_type: str = "industry",
    ) -> None:
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.board_type = board_type

    def predict(self, top_n: int = 10) -> pd.DataFrame:
        """预测明日热门板块。

        Parameters
        ----------
        top_n : int
            返回前 N 个推荐板块。

        Returns
        -------
        pd.DataFrame
            包含板块名称、综合评分、各因子得分、涨跌幅、资金流向等信息。
        """
        logger.info("开始计算板块热度评分...")

        # 1. 获取板块行情
        board_df = SectorMonitor.get_board_list(self.board_type)

        # 2. 获取资金流向
        sector_type = "行业资金流" if self.board_type == "industry" else "概念资金流"
        try:
            fund_df = SectorMonitor.get_fund_flow(
                period="今日", sector_type=sector_type,
            )
        except Exception:
            logger.warning("资金流向数据获取失败，使用零值替代")
            fund_df = pd.DataFrame()

        # 3. 合并数据
        merged = self._merge_data(board_df, fund_df)

        # 4. 计算各因子得分
        merged = self._score_fund_flow(merged)
        merged = self._score_momentum(merged)
        merged = self._score_activity(merged)
        merged = self._score_breadth(merged)
        merged = self._score_lead_strength(merged)

        # 5. 计算综合评分
        merged["total_score"] = (
            self.weights["fund_flow"] * merged["score_fund_flow"]
            + self.weights["momentum"] * merged["score_momentum"]
            + self.weights["activity"] * merged["score_activity"]
            + self.weights["breadth"] * merged["score_breadth"]
            + self.weights["lead_strength"] * merged["score_lead_strength"]
        )

        # 6. 排序并返回
        result = merged.nlargest(top_n, "total_score")

        output_cols = [
            "name", "code", "total_score",
            "change_pct", "turnover_rate",
            "rise_count", "fall_count",
            "lead_stock", "lead_stock_pct",
            "net_inflow",
            "score_fund_flow", "score_momentum",
            "score_activity", "score_breadth", "score_lead_strength",
        ]
        available_cols = [c for c in output_cols if c in result.columns]
        return result[available_cols].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 数据合并
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_data(board_df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
        """合并板块行情与资金流向数据。"""
        df = board_df.copy()

        if fund_df.empty:
            df["net_inflow"] = 0.0
            df["net_inflow_pct"] = 0.0
            return df

        # 资金流向数据的列名可能包含"名称"
        fund_name_col = None
        for col in ["名称", "name"]:
            if col in fund_df.columns:
                fund_name_col = col
                break

        if fund_name_col is None:
            df["net_inflow"] = 0.0
            df["net_inflow_pct"] = 0.0
            return df

        # 查找净流入列
        inflow_col = None
        inflow_pct_col = None
        for col in fund_df.columns:
            if "主力净流入-净额" in col:
                inflow_col = col
            if "主力净流入-净占比" in col:
                inflow_pct_col = col

        fund_subset = fund_df[[fund_name_col]].copy()
        fund_subset = fund_subset.rename(columns={fund_name_col: "name"})

        if inflow_col:
            fund_subset["net_inflow"] = fund_df[inflow_col].values
        else:
            fund_subset["net_inflow"] = 0.0

        if inflow_pct_col:
            fund_subset["net_inflow_pct"] = fund_df[inflow_pct_col].values
        else:
            fund_subset["net_inflow_pct"] = 0.0

        df = df.merge(fund_subset, on="name", how="left")
        df["net_inflow"] = df["net_inflow"].fillna(0.0)
        df["net_inflow_pct"] = df["net_inflow_pct"].fillna(0.0)

        return df

    # ------------------------------------------------------------------
    # 因子评分（0-100 标准化）
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        """Min-Max 标准化到 0-100。"""
        smin = series.min()
        smax = series.max()
        if smax == smin:
            return pd.Series(50.0, index=series.index)
        return ((series - smin) / (smax - smin)) * 100

    def _score_fund_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """资金流向因子：主力净流入越多越好。"""
        df["score_fund_flow"] = self._normalize(df["net_inflow"])
        return df

    def _score_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """动量因子：当日涨幅适中为佳（过高可能追高风险）。

        使用涨幅排名百分位，偏好涨幅在 top20%-60% 区间的板块。
        """
        pct_rank = df["change_pct"].rank(pct=True)
        # 偏好中等偏上涨幅：越接近 0.7 分越高
        df["score_momentum"] = self._normalize(
            100 - ((pct_rank - 0.7).abs() * 200)
        )
        return df

    def _score_activity(self, df: pd.DataFrame) -> pd.DataFrame:
        """活跃度因子：换手率越高越活跃。"""
        df["score_activity"] = self._normalize(df["turnover_rate"])
        return df

    def _score_breadth(self, df: pd.DataFrame) -> pd.DataFrame:
        """上涨广度因子：板块内上涨家数占比越高越好。"""
        total = df["rise_count"] + df["fall_count"]
        total = total.replace(0, 1)  # 避免除零
        breadth = df["rise_count"] / total
        df["score_breadth"] = self._normalize(breadth)
        return df

    def _score_lead_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """领涨强度因子：领涨股涨幅越高，板块趋势越强。"""
        df["score_lead_strength"] = self._normalize(df["lead_stock_pct"])
        return df
