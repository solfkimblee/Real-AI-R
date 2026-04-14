"""热门板块股票推荐模块

获取指定板块的成分股，并基于多维度排序推荐优质个股：
- 涨幅排名（技术面）
- 换手率（活跃度）
- 量比（资金关注度）
- 市盈率 / 市净率（估值合理性）
- 营收增长率（成长性）— 基本面因子
- 研发投入占比（创新能力）— 基本面因子
"""

from __future__ import annotations

import logging

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)


class StockRecommender:
    """板块内个股推荐器。"""

    @staticmethod
    def get_board_stocks(
        board_name: str,
        board_type: str = "industry",
    ) -> pd.DataFrame:
        """获取板块成分股列表及实时行情。

        Parameters
        ----------
        board_name : str
            板块名称，如 "半导体"、"人工智能"。
        board_type : str
            "industry" | "concept"

        Returns
        -------
        pd.DataFrame
            标准化列：code, name, price, change_pct, volume, amount,
            turnover_rate, pe_dynamic, amplitude 等。
        """
        logger.info("获取板块 [%s] 成分股", board_name)

        if board_type == "industry":
            df = ak.stock_board_industry_cons_em(symbol=board_name)
        else:
            df = ak.stock_board_concept_cons_em(symbol=board_name)

        col_map = {
            "序号": "rank",
            "代码": "code",
            "名称": "name",
            "最新价": "price",
            "涨跌幅": "change_pct",
            "涨跌额": "change_amount",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "最高": "high",
            "最低": "low",
            "今开": "open",
            "昨收": "prev_close",
            "换手率": "turnover_rate",
            "市盈率-动态": "pe_dynamic",
            "市净率": "pb",
        }
        df = df.rename(columns=col_map)
        return df

    @staticmethod
    def get_fundamental_data(codes: list[str]) -> pd.DataFrame:
        """批量获取个股基本面数据（营收增长率、研发投入等）。

        使用 AKShare 的 stock_financial_abstract 接口获取最近财报数据。
        为了避免过多 API 调用，对列表做截断。

        Parameters
        ----------
        codes : list[str]
            股票代码列表。

        Returns
        -------
        pd.DataFrame
            含 code, revenue_growth, rd_ratio 等列。
        """
        records = []
        # 限制最多查询 30 只以避免 API 限流
        for code in codes[:30]:
            try:
                df = ak.stock_financial_abstract(symbol=code)
                if df.empty:
                    continue
                # 提取最近两个完整年度的营收总收入（行索引1）
                row_revenue = df[df["选项"] == "常用指标"]
                if row_revenue.empty:
                    continue
                # 指标列：按列名（日期格式 YYYYMMDD）取最新的年报
                date_cols = [
                    c for c in df.columns
                    if c not in ("选项", "指标") and c.endswith("1231")
                ]
                date_cols = sorted(date_cols, reverse=True)

                revenue_growth = None
                if len(date_cols) >= 2:
                    # 营业总收入行
                    rev_row = df[df["指标"] == "营业总收入"]
                    if not rev_row.empty:
                        latest = pd.to_numeric(
                            rev_row.iloc[0][date_cols[0]], errors="coerce",
                        )
                        prev = pd.to_numeric(
                            rev_row.iloc[0][date_cols[1]], errors="coerce",
                        )
                        if pd.notna(latest) and pd.notna(prev) and prev > 0:
                            revenue_growth = (latest - prev) / prev * 100

                records.append({
                    "code": code,
                    "revenue_growth": revenue_growth,
                })
            except Exception:
                logger.debug("获取 %s 基本面数据失败", code)
                continue

        if not records:
            return pd.DataFrame(columns=["code", "revenue_growth"])
        return pd.DataFrame(records)

    @staticmethod
    def recommend(
        board_name: str,
        board_type: str = "industry",
        top_n: int = 10,
        sort_by: str = "composite",
        include_fundamental: bool = False,
    ) -> pd.DataFrame:
        """推荐板块内的优质个股。

        Parameters
        ----------
        board_name : str
            板块名称。
        board_type : str
            "industry" | "concept"
        top_n : int
            推荐个股数量。
        sort_by : str
            排序方式：
            - "composite": 综合评分（默认）
            - "change_pct": 按涨幅
            - "turnover_rate": 按换手率
            - "amount": 按成交额
        include_fundamental : bool
            是否加入基本面因子（PE/PB/营收增长）。
            开启后会额外请求财务数据，速度较慢。

        Returns
        -------
        pd.DataFrame
            推荐个股列表，含评分及各维度排名。
        """
        df = StockRecommender.get_board_stocks(board_name, board_type)

        if df.empty:
            return df

        # 过滤掉 ST 股和停牌股
        df = df[~df["name"].str.contains("ST|退市", na=False)].copy()
        df = df[df["price"].notna() & (df["price"] > 0)].copy()

        if df.empty:
            return df

        if sort_by != "composite":
            if sort_by in df.columns:
                return df.nlargest(top_n, sort_by).reset_index(drop=True)
            return df.head(top_n).reset_index(drop=True)

        # 综合评分
        if include_fundamental:
            df = StockRecommender._compute_composite_score_with_fundamental(df)
        else:
            df = StockRecommender._compute_composite_score(df)

        result = df.nlargest(top_n, "composite_score")

        output_cols = [
            "code", "name", "price", "change_pct",
            "turnover_rate", "amount", "pe_dynamic", "pb",
            "revenue_growth", "composite_score",
        ]
        available = [c for c in output_cols if c in result.columns]
        return result[available].reset_index(drop=True)

    @staticmethod
    def _compute_composite_score(df: pd.DataFrame) -> pd.DataFrame:
        """计算个股综合评分（技术面）。

        评分维度（满分100）：
        - 涨幅排名 (30%)：涨幅越高越好
        - 换手率排名 (25%)：适度活跃
        - 成交额排名 (25%)：资金关注度
        - 估值合理性 (20%)：PE 适中（排除负值）
        """
        df = df.copy()

        # 涨幅得分
        df["s_change"] = _normalize(df["change_pct"])

        # 换手率得分
        if "turnover_rate" in df.columns:
            df["s_turnover"] = _normalize(df["turnover_rate"])
        else:
            df["s_turnover"] = 50.0

        # 成交额得分
        if "amount" in df.columns:
            df["s_amount"] = _normalize(df["amount"])
        else:
            df["s_amount"] = 50.0

        # 估值得分：PE 为正且合理区间得分高
        if "pe_dynamic" in df.columns:
            pe = df["pe_dynamic"].copy()
            # 负值或极大值视为不佳
            pe = pe.where((pe > 0) & (pe < 500), other=500)
            # PE 越低越好（取反标准化）
            df["s_valuation"] = _normalize(-pe)
        else:
            df["s_valuation"] = 50.0

        df["composite_score"] = (
            0.30 * df["s_change"]
            + 0.25 * df["s_turnover"]
            + 0.25 * df["s_amount"]
            + 0.20 * df["s_valuation"]
        )

        return df

    @staticmethod
    def _compute_composite_score_with_fundamental(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """计算含基本面因子的综合评分。

        评分维度（满分100）：
        - 涨幅 (20%)
        - 换手率 (15%)
        - 成交额 (15%)
        - PE估值 (15%) — PE越低越好
        - PB估值 (10%) — PB越低越好
        - 营收增长率 (25%) — 增长越快越好
        """
        df = df.copy()

        # 技术面得分
        df["s_change"] = _normalize(df["change_pct"])

        if "turnover_rate" in df.columns:
            df["s_turnover"] = _normalize(df["turnover_rate"])
        else:
            df["s_turnover"] = 50.0

        if "amount" in df.columns:
            df["s_amount"] = _normalize(df["amount"])
        else:
            df["s_amount"] = 50.0

        # PE 估值得分
        if "pe_dynamic" in df.columns:
            pe = df["pe_dynamic"].copy()
            pe = pe.where((pe > 0) & (pe < 500), other=500)
            df["s_pe"] = _normalize(-pe)
        else:
            df["s_pe"] = 50.0

        # PB 估值得分
        if "pb" in df.columns:
            pb = df["pb"].copy()
            pb = pb.where((pb > 0) & (pb < 50), other=50)
            df["s_pb"] = _normalize(-pb)
        else:
            df["s_pb"] = 50.0

        # 营收增长率 — 从财务数据获取
        codes = df["code"].tolist()
        fund_df = StockRecommender.get_fundamental_data(codes)

        if not fund_df.empty:
            df = df.merge(fund_df, on="code", how="left")
            rev = df["revenue_growth"].copy()
            rev = rev.fillna(0)
            # 限制极端值
            rev = rev.clip(-100, 500)
            df["s_revenue_growth"] = _normalize(rev)
        else:
            df["revenue_growth"] = None
            df["s_revenue_growth"] = 50.0

        df["composite_score"] = (
            0.20 * df["s_change"]
            + 0.15 * df["s_turnover"]
            + 0.15 * df["s_amount"]
            + 0.15 * df["s_pe"]
            + 0.10 * df["s_pb"]
            + 0.25 * df["s_revenue_growth"]
        )

        return df


def _normalize(series: pd.Series) -> pd.Series:
    """Min-Max 标准化到 0-100。"""
    smin = series.min()
    smax = series.max()
    if smax == smin:
        return pd.Series(50.0, index=series.index)
    return ((series - smin) / (smax - smin)) * 100
