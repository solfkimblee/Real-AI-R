"""催化剂追踪模块

追踪影响投资决策的关键催化剂事件：
- 财报日历：业绩预告、业绩快报、财报披露日期
- 政策事件：宏观政策动态
- 产业新闻：板块/个股相关新闻动态
"""

from __future__ import annotations

import logging
from datetime import datetime

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)


class CatalystTracker:
    """催化剂事件追踪器。"""

    # 财报披露季节（季报截止日期 → 通常披露窗口）
    REPORT_SEASONS = {
        "一季报": {"deadline": "0331", "window": ("04-01", "04-30")},
        "中报": {"deadline": "0630", "window": ("07-01", "08-31")},
        "三季报": {"deadline": "0930", "window": ("10-01", "10-31")},
        "年报": {"deadline": "1231", "window": ("01-01", "04-30")},
    }

    @staticmethod
    def get_earnings_forecast(period: str = "20250331") -> pd.DataFrame:
        """获取业绩预告数据。

        Parameters
        ----------
        period : str
            报告期，格式 YYYYMMDD，如 "20250331" 表示2025年一季报。

        Returns
        -------
        pd.DataFrame
            业绩预告列表。
        """
        try:
            df = ak.stock_yjyg_em(date=period)
            if df.empty:
                return pd.DataFrame()

            col_map = {
                "序号": "rank",
                "股票代码": "code",
                "股票简称": "name",
                "预测指标": "indicator",
                "业绩变动": "change_desc",
                "预测数值": "forecast_value",
                "业绩变动幅度": "change_pct",
                "业绩变动原因": "reason",
                "预告类型": "forecast_type",
                "上年同期值": "prev_value",
                "公告日期": "announce_date",
            }
            df = df.rename(columns=col_map)
            return df
        except Exception as e:
            logger.warning("获取业绩预告失败: %s", e)
            return pd.DataFrame()

    @staticmethod
    def get_earnings_express(period: str = "20250331") -> pd.DataFrame:
        """获取业绩快报数据。

        Parameters
        ----------
        period : str
            报告期，格式 YYYYMMDD。

        Returns
        -------
        pd.DataFrame
            业绩快报列表。
        """
        try:
            df = ak.stock_yjkb_em(date=period)
            if df.empty:
                return pd.DataFrame()

            col_map = {
                "序号": "rank",
                "股票代码": "code",
                "股票简称": "name",
                "每股收益": "eps",
                "营业收入-营业收入": "revenue",
                "营业收入-去年同期": "revenue_prev",
                "营业收入-同比增长": "revenue_yoy",
                "营业收入-季度环比增长": "revenue_qoq",
                "净利润-净利润": "net_profit",
                "净利润-去年同期": "profit_prev",
                "净利润-同比增长": "profit_yoy",
                "净利润-季度环比增长": "profit_qoq",
                "每股净资产": "bps",
                "净资产收益率": "roe",
                "所处行业": "industry",
                "公告日期": "announce_date",
            }
            df = df.rename(columns=col_map)
            return df
        except Exception as e:
            logger.warning("获取业绩快报失败: %s", e)
            return pd.DataFrame()

    @staticmethod
    def get_current_report_period() -> str:
        """根据当前日期推算最近的财报期。"""
        now = datetime.now()
        year = now.year
        month = now.month

        if month <= 4:
            # 一季度：上年年报或当年一季报
            return f"{year}0331"
        elif month <= 8:
            return f"{year}0630"
        elif month <= 10:
            return f"{year}0930"
        else:
            return f"{year}1231"

    @staticmethod
    def get_report_calendar() -> list[dict]:
        """获取当前的财报日历信息。"""
        now = datetime.now()
        year = now.year
        calendar = []

        for name, info in CatalystTracker.REPORT_SEASONS.items():
            deadline = info["deadline"]
            window_start, window_end = info["window"]

            # 年报跨年处理
            if name == "年报":
                report_year = year - 1 if now.month <= 4 else year
                period = f"{report_year}{deadline}"
                disc_start = f"{report_year + 1}-{window_start}"
                disc_end = f"{report_year + 1}-{window_end}"
            else:
                period = f"{year}{deadline}"
                disc_start = f"{year}-{window_start}"
                disc_end = f"{year}-{window_end}"

            try:
                disc_start_dt = datetime.strptime(disc_start, "%Y-%m-%d")
                disc_end_dt = datetime.strptime(disc_end, "%Y-%m-%d")
                if now < disc_start_dt:
                    status = "未开始"
                elif now > disc_end_dt:
                    status = "已结束"
                else:
                    status = "进行中"
            except ValueError:
                status = "未知"

            calendar.append({
                "report_name": name,
                "period": period,
                "disclosure_window": f"{disc_start} ~ {disc_end}",
                "status": status,
            })

        return calendar

    @staticmethod
    def get_macro_events() -> list[dict]:
        """获取近期重要宏观政策事件。

        由于 AKShare 没有直接的政策日历 API，这里返回结构化的
        宏观事件框架供用户参考。
        """
        events = []

        # 基于已知的定期经济数据发布节奏
        macro_schedule = [
            {
                "event": "CPI/PPI数据发布",
                "frequency": "每月中旬",
                "impact": "通胀走势影响货币政策",
                "sectors": "贵金属、消费、银行",
            },
            {
                "event": "PMI数据发布",
                "frequency": "每月1日",
                "impact": "制造业景气度，经济先行指标",
                "sectors": "工业、制造业、基本金属",
            },
            {
                "event": "社融/M2数据",
                "frequency": "每月中旬",
                "impact": "流动性松紧，信用环境",
                "sectors": "银行、地产、科技成长",
            },
            {
                "event": "GDP数据发布",
                "frequency": "每季度中旬",
                "impact": "经济增速趋势判断",
                "sectors": "全市场",
            },
            {
                "event": "LPR报价",
                "frequency": "每月20日",
                "impact": "贷款基准利率变化",
                "sectors": "银行、地产、消费",
            },
            {
                "event": "中央政治局会议",
                "frequency": "每季度末",
                "impact": "政策定调与方向",
                "sectors": "全市场",
            },
            {
                "event": "美联储议息会议(FOMC)",
                "frequency": "每6周一次",
                "impact": "全球流动性、汇率、外资流向",
                "sectors": "科技、贵金属、出口链",
            },
            {
                "event": "国务院常务会议",
                "frequency": "每周三",
                "impact": "具体产业政策出台",
                "sectors": "被提及的具体行业",
            },
        ]

        for item in macro_schedule:
            events.append({
                "event": item["event"],
                "frequency": item["frequency"],
                "impact": item["impact"],
                "affected_sectors": item["sectors"],
                "category": "宏观政策",
            })

        return events

    @staticmethod
    def get_industry_catalysts() -> list[dict]:
        """获取产业级催化剂事件框架。

        基于泽平方法论的六大科技赛道和周期五段论，
        列出各赛道的关键催化剂。
        """
        catalysts = [
            # 科技主线催化剂
            {
                "track": "芯片/半导体/算力",
                "category": "科技主线",
                "catalysts": [
                    "国产大算力芯片流片/量产消息",
                    "算力中心建设招标公告",
                    "半导体设备国产替代突破",
                    "英伟达/AMD新品发布（竞品催化）",
                ],
            },
            {
                "track": "大模型/Agent应用",
                "category": "科技主线",
                "catalysts": [
                    "国产大模型新版本发布",
                    "Agent应用日活/收入里程碑",
                    "头部企业AI战略发布会",
                    "AI应用商业化落地案例",
                ],
            },
            {
                "track": "机器人/自动驾驶",
                "category": "科技主线",
                "catalysts": [
                    "人形机器人量产/出货数据",
                    "自动驾驶路测牌照发放",
                    "特斯拉/宇树机器人新进展",
                    "智驾渗透率数据发布",
                ],
            },
            {
                "track": "AI医疗/商业航天",
                "category": "科技主线",
                "catalysts": [
                    "AI新药获批临床/上市",
                    "商业航天发射计划",
                    "低轨卫星组网进展",
                    "创新药出海授权交易",
                ],
            },
            # 周期主线催化剂
            {
                "track": "贵金属（金银）",
                "category": "周期主线",
                "catalysts": [
                    "美联储降息/鸽派发言",
                    "地缘冲突升级",
                    "各国央行购金数据",
                    "美元指数大幅走弱",
                ],
            },
            {
                "track": "基本金属（铜铝）",
                "category": "周期主线",
                "catalysts": [
                    "新能源车销量超预期",
                    "电网投资计划发布",
                    "铜矿供应中断",
                    "库存创新低数据",
                ],
            },
            {
                "track": "传统能源（煤炭石油）",
                "category": "周期主线",
                "catalysts": [
                    "OPEC+减产决议",
                    "中东局势变化",
                    "极端天气影响供应",
                    "战略储备释放/补库消息",
                ],
            },
            {
                "track": "农业后周期",
                "category": "周期主线",
                "catalysts": [
                    "厄尔尼诺/拉尼娜预警",
                    "化肥价格上涨",
                    "转基因商业化政策",
                    "粮食安全相关政策",
                ],
            },
        ]
        return catalysts
