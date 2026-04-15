"""避雷过滤器 — 红线禁区板块与个股过滤

基于泽平宏观框架定义的四大"红线禁区"：
1. 房地产及泛地产链 — 人口结构与城镇化见顶
2. 传统白酒 — 商务宴请需求塌方
3. 传统零售/低端餐饮 — 消费降级挤压
4. 旧软件外包/客服 — Agent 无情清洗

提供板块级和个股级两层过滤能力。
"""

from __future__ import annotations

import logging

import pandas as pd

from real_ai_r.macro.classifier import REDLINE_ZONES

logger = logging.getLogger(__name__)


class RedLineFilter:
    """红线禁区过滤器。

    支持对板块列表和个股列表进行过滤，自动剔除属于
    红线禁区的标的。
    """

    def __init__(self, redline_zones: dict | None = None) -> None:
        self.zones = redline_zones or REDLINE_ZONES
        self._keywords = self._collect_keywords()

    def _collect_keywords(self) -> list[str]:
        """收集所有红线关键词。"""
        keywords = []
        for zone in self.zones.values():
            keywords.extend(zone["keywords"])
        return keywords

    def is_redline(self, name: str) -> bool:
        """判断板块/股票是否属于红线禁区。

        Parameters
        ----------
        name : str
            板块名称或行业标签。

        Returns
        -------
        bool
            True 表示属于红线禁区。
        """
        for kw in self._keywords:
            if kw in name or name in kw:
                return True
        return False

    def get_redline_reason(self, name: str) -> str | None:
        """获取某板块被标记为红线的原因。

        Returns
        -------
        str | None
            红线原因描述，不在红线内返回 None。
        """
        for zone_info in self.zones.values():
            for kw in zone_info["keywords"]:
                if kw in name or name in kw:
                    return f"{zone_info['icon']} {zone_info['display']}：{zone_info['desc']}"
        return None

    def filter_boards(
        self,
        df: pd.DataFrame,
        name_col: str = "name",
        keep_redline: bool = False,
    ) -> pd.DataFrame:
        """过滤板块 DataFrame，剔除红线禁区板块。

        Parameters
        ----------
        df : pd.DataFrame
            板块数据。
        name_col : str
            板块名称列。
        keep_redline : bool
            如果为 True，则返回红线板块（反向过滤）。

        Returns
        -------
        pd.DataFrame
            过滤后的 DataFrame。
        """
        mask = df[name_col].apply(lambda n: self.is_redline(str(n)))

        if keep_redline:
            result = df[mask].copy()
        else:
            result = df[~mask].copy()

        filtered_count = mask.sum()
        logger.info(
            "红线过滤: 共 %d 个板块, 过滤 %d 个, 保留 %d 个",
            len(df), filtered_count, len(result),
        )

        return result

    def filter_stocks(
        self,
        df: pd.DataFrame,
        industry_col: str | None = None,
        name_col: str = "name",
    ) -> pd.DataFrame:
        """过滤个股 DataFrame。

        同时检查股票名称中的 ST/退市标记，以及行业分类标签。

        Parameters
        ----------
        df : pd.DataFrame
            个股数据。
        industry_col : str | None
            行业列名。如果提供，同时按行业过滤。
        name_col : str
            股票名称列。

        Returns
        -------
        pd.DataFrame
            过滤后的 DataFrame。
        """
        df = df.copy()

        # 基础 ST/退市 过滤
        if name_col in df.columns:
            df = df[~df[name_col].str.contains("ST|退市", na=False)]

        # 行业红线过滤
        if industry_col and industry_col in df.columns:
            df = df[~df[industry_col].apply(lambda n: self.is_redline(str(n)))]

        return df

    def get_redline_summary(self, df: pd.DataFrame, name_col: str = "name") -> list[dict]:
        """获取红线板块的详细摘要。

        Returns
        -------
        list[dict]
            每个红线板块的信息，包含名称、分类、原因等。
        """
        results = []
        for _, row in df.iterrows():
            name = str(row[name_col])
            if self.is_redline(name):
                reason = self.get_redline_reason(name)
                entry = {"name": name, "reason": reason or "红线禁区"}
                if "change_pct" in df.columns:
                    entry["change_pct"] = row["change_pct"]
                if "turnover_rate" in df.columns:
                    entry["turnover_rate"] = row["turnover_rate"]
                results.append(entry)
        return results

    @staticmethod
    def get_zone_descriptions() -> list[dict]:
        """获取所有红线禁区的描述信息。"""
        return [
            {
                "name": zone["display"],
                "icon": zone["icon"],
                "description": zone["desc"],
                "keywords": zone["keywords"],
            }
            for zone in REDLINE_ZONES.values()
        ]
