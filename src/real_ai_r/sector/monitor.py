"""板块监控模块 — 实时板块行情、资金流向、涨跌统计

提供行业板块和概念板块的实时监控能力。
"""

from __future__ import annotations

import logging
from typing import Literal

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)

BoardType = Literal["industry", "concept"]


class SectorMonitor:
    """板块监控器 — 获取实时板块行情与资金流向。"""

    # ------------------------------------------------------------------
    # 板块行情
    # ------------------------------------------------------------------

    @staticmethod
    def get_board_list(board_type: BoardType = "industry") -> pd.DataFrame:
        """获取板块列表及实时行情。

        Parameters
        ----------
        board_type : "industry" | "concept"
            板块类型：行业板块 或 概念板块。

        Returns
        -------
        pd.DataFrame
            标准化列：rank, name, code, price, change_pct, total_mv,
            turnover_rate, rise_count, fall_count, lead_stock, lead_stock_pct
        """
        logger.info("获取%s板块列表", "行业" if board_type == "industry" else "概念")

        if board_type == "industry":
            df = ak.stock_board_industry_name_em()
        else:
            df = ak.stock_board_concept_name_em()

        col_map = {
            "排名": "rank",
            "板块名称": "name",
            "板块代码": "code",
            "最新价": "price",
            "涨跌额": "change_amount",
            "涨跌幅": "change_pct",
            "总市值": "total_mv",
            "换手率": "turnover_rate",
            "上涨家数": "rise_count",
            "下跌家数": "fall_count",
            "领涨股票": "lead_stock",
            "领涨股票-涨跌幅": "lead_stock_pct",
        }
        df = df.rename(columns=col_map)
        return df

    # ------------------------------------------------------------------
    # 资金流向
    # ------------------------------------------------------------------

    @staticmethod
    def get_fund_flow(
        period: str = "今日",
        sector_type: str = "行业资金流",
    ) -> pd.DataFrame:
        """获取板块资金流向排名。

        Parameters
        ----------
        period : str
            时间周期："今日" | "5日" | "10日"
        sector_type : str
            "行业资金流" | "概念资金流"

        Returns
        -------
        pd.DataFrame
            包含主力净流入、超大单、大单、中单、小单流向数据。
        """
        logger.info("获取板块资金流向: %s - %s", sector_type, period)
        df = ak.stock_sector_fund_flow_rank(
            indicator=period,
            sector_type=sector_type,
        )

        # 标准化列名（去除时间前缀）
        new_cols = {}
        for col in df.columns:
            clean = col
            for prefix in ["今日", "5日", "10日"]:
                clean = clean.replace(prefix, "")
            new_cols[col] = clean.strip()
        df = df.rename(columns=new_cols)

        return df

    # ------------------------------------------------------------------
    # 板块涨跌统计
    # ------------------------------------------------------------------

    @staticmethod
    def get_board_stats(board_type: BoardType = "industry") -> dict:
        """获取板块涨跌统计摘要。

        Returns
        -------
        dict
            包含 rise_count, fall_count, flat_count, top_boards, bottom_boards
        """
        df = SectorMonitor.get_board_list(board_type)

        rise = df[df["change_pct"] > 0]
        fall = df[df["change_pct"] < 0]
        flat = df[df["change_pct"] == 0]

        return {
            "rise_count": len(rise),
            "fall_count": len(fall),
            "flat_count": len(flat),
            "total_count": len(df),
            "top5": df.nlargest(5, "change_pct")[
                ["name", "change_pct", "lead_stock", "lead_stock_pct"]
            ].to_dict("records"),
            "bottom5": df.nsmallest(5, "change_pct")[
                ["name", "change_pct", "lead_stock", "lead_stock_pct"]
            ].to_dict("records"),
            "avg_change": df["change_pct"].mean(),
        }
