"""历史数据采集模块

从 AKShare 采集板块历史行情数据，构建训练数据集。
支持行业板块和概念板块的日级别历史数据。
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import akshare as ak
import pandas as pd

from real_ai_r.sector.monitor import SectorMonitor

logger = logging.getLogger(__name__)


class BoardHistoryCollector:
    """板块历史数据采集器。

    采集板块日线数据（OHLCV + 涨跌幅）用于特征工程和模型训练。
    """

    # AKShare 限流：每个请求之间加间隔避免被封
    REQUEST_DELAY = 0.3

    def __init__(self, board_type: str = "industry") -> None:
        self.board_type = board_type

    def collect_board_history(
        self,
        board_name: str,
        days: int = 60,
    ) -> pd.DataFrame:
        """采集单个板块的历史日线数据。

        Parameters
        ----------
        board_name : str
            板块名称。
        days : int
            历史天数。

        Returns
        -------
        pd.DataFrame
            标准化列：date, open, close, high, low, volume, change_pct, turnover_rate, amplitude
        """
        logger.info("采集板块 [%s] 最近 %d 天历史", board_name, days)
        try:
            if self.board_type == "industry":
                df = ak.stock_board_industry_hist_em(
                    symbol=board_name,
                    period="日k",
                    adjust="",
                )
            else:
                df = ak.stock_board_concept_hist_em(
                    symbol=board_name,
                    period="日k",
                    adjust="",
                )
        except Exception:
            logger.warning("板块 [%s] 历史数据获取失败", board_name)
            return pd.DataFrame()

        if df.empty:
            return df

        col_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "change_pct",
            "涨跌额": "change_amount",
            "换手率": "turnover_rate",
        }
        df = df.rename(columns=col_map)

        # 确保日期排序
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

        # 截取最近 N 天
        if len(df) > days:
            df = df.tail(days).reset_index(drop=True)

        df["board_name"] = board_name
        return df

    def collect_all_boards(
        self,
        days: int = 60,
        max_boards: int = 0,
    ) -> pd.DataFrame:
        """采集所有板块的历史数据。

        Parameters
        ----------
        days : int
            历史天数。
        max_boards : int
            最大采集板块数。0 表示全部。

        Returns
        -------
        pd.DataFrame
            拼接后的所有板块历史数据。
        """
        logger.info("开始批量采集板块历史数据...")
        try:
            board_list = SectorMonitor.get_board_list(self.board_type)
        except Exception:
            logger.error("板块列表获取失败")
            return pd.DataFrame()

        board_names = board_list["name"].tolist()
        if max_boards > 0:
            board_names = board_names[:max_boards]

        all_data = []
        for i, name in enumerate(board_names):
            logger.info("进度: %d/%d — %s", i + 1, len(board_names), name)
            df = self.collect_board_history(name, days=days)
            if not df.empty:
                all_data.append(df)
            time.sleep(self.REQUEST_DELAY)

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)


class SnapshotCollector:
    """板块截面快照采集器。

    采集当日板块行情截面数据，用于实时预测。
    """

    @staticmethod
    def collect_today_snapshot(board_type: str = "industry") -> pd.DataFrame:
        """采集今日板块截面快照。

        Returns
        -------
        pd.DataFrame
            包含板块名称、涨跌幅、换手率、资金流向等截面数据。
        """
        try:
            board_df = SectorMonitor.get_board_list(board_type)
        except Exception:
            logger.error("板块列表获取失败")
            return pd.DataFrame()

        # 尝试获取资金流向
        sector_type = "行业资金流" if board_type == "industry" else "概念资金流"
        try:
            fund_df = SectorMonitor.get_fund_flow("今日", sector_type)
        except Exception:
            fund_df = pd.DataFrame()

        # 合并资金流向
        if not fund_df.empty:
            name_col = None
            for col in ["名称", "name"]:
                if col in fund_df.columns:
                    name_col = col
                    break

            if name_col:
                inflow_col = None
                for col in fund_df.columns:
                    if "主力净流入-净额" in col:
                        inflow_col = col
                        break

                fund_subset = fund_df[[name_col]].copy()
                fund_subset = fund_subset.rename(columns={name_col: "name"})
                if inflow_col:
                    fund_subset["net_inflow"] = fund_df[inflow_col].values
                else:
                    fund_subset["net_inflow"] = 0.0

                board_df = board_df.merge(fund_subset, on="name", how="left")
                board_df["net_inflow"] = board_df["net_inflow"].fillna(0.0)
            else:
                board_df["net_inflow"] = 0.0
        else:
            board_df["net_inflow"] = 0.0

        board_df["snapshot_date"] = datetime.now().strftime("%Y-%m-%d")
        return board_df
