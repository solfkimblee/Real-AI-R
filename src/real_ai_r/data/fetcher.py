"""数据获取模块 — AKShare 封装

提供 A股日线、分钟线、财务数据获取能力。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)

# 本地缓存目录
_CACHE_DIR = Path("data_cache")


class DataFetcher:
    """A股数据获取器，基于 AKShare。

    支持：
    - 日线行情（前/后复权）
    - 实时行情快照
    - 股票列表
    - 本地 CSV 缓存（可选）
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 日线行情
    # ------------------------------------------------------------------

    def get_stock_daily(
        self,
        symbol: str,
        start_date: str = "20200101",
        end_date: str | None = None,
        adjust: str = "qfq",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """获取个股日线行情数据。

        Parameters
        ----------
        symbol : str
            股票代码，如 "000001"（平安银行）、"600519"（贵州茅台）。
        start_date : str
            开始日期，格式 YYYYMMDD 或 YYYY-MM-DD。
        end_date : str | None
            结束日期，默认今天。
        adjust : str
            复权方式："qfq"（前复权） | "hfq"（后复权） | ""（不复权）。
        use_cache : bool
            是否使用本地缓存。

        Returns
        -------
        pd.DataFrame
            标准化列名：date, open, high, low, close, volume, amount, turnover
        """
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")

        cache_key = f"{symbol}_{start_date}_{end_date}_{adjust}"
        cache_path = self.cache_dir / f"{cache_key}.csv"

        if use_cache and cache_path.exists():
            logger.info("从缓存加载: %s", cache_path)
            df = pd.read_csv(cache_path, parse_dates=["date"])
            return df

        logger.info("从 AKShare 获取: %s (%s ~ %s)", symbol, start_date, end_date)
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )

        df = self._standardize_daily(df)

        if use_cache:
            df.to_csv(cache_path, index=False)
            logger.info("已缓存: %s", cache_path)

        return df

    def get_index_daily(
        self,
        symbol: str = "000300",
        start_date: str = "20200101",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """获取指数日线行情（如沪深300、上证指数）。

        Parameters
        ----------
        symbol : str
            指数代码，如 "000300"（沪深300）、"000001"（上证指数）。
        """
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")

        logger.info("获取指数行情: %s", symbol)
        df = ak.stock_zh_index_daily_em(symbol=f"sh{symbol}")

        df = df.rename(
            columns={
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= pd.to_datetime(start_date)) & (
            df["date"] <= pd.to_datetime(end_date)
        )
        df = df.loc[mask].reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # 股票列表
    # ------------------------------------------------------------------

    def get_stock_list(self) -> pd.DataFrame:
        """获取A股全市场股票列表。"""
        logger.info("获取A股股票列表")
        df = ak.stock_zh_a_spot_em()
        df = df[["代码", "名称", "最新价", "涨跌幅", "总市值", "流通市值"]].copy()
        df.columns = ["code", "name", "price", "change_pct", "market_cap", "float_cap"]
        return df

    # ------------------------------------------------------------------
    # 实时行情
    # ------------------------------------------------------------------

    def get_realtime_quote(self, symbol: str) -> pd.Series:
        """获取单只股票实时行情。"""
        df = ak.stock_zh_a_spot_em()
        row = df[df["代码"] == symbol]
        if row.empty:
            raise ValueError(f"未找到股票: {symbol}")
        return row.iloc[0]

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _standardize_daily(df: pd.DataFrame) -> pd.DataFrame:
        """标准化 AKShare 日线数据列名。"""
        col_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
            "涨跌幅": "change_pct",
            "涨跌额": "change",
            "振幅": "amplitude",
        }
        df = df.rename(columns=col_map)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # 确保核心列存在
        for col in ["date", "open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                raise ValueError(f"缺少必要列: {col}")

        return df
