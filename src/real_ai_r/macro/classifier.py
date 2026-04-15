"""板块分类标签系统

基于泽平宏观框架，将A股板块划分为三大类别：
- 科技主线（康波第六波）：芯片/半导体、大模型/Agent、机器人/自动驾驶、AI医疗、商业航天
- 周期主线（大宗商品五段论）：贵金属、基本金属、传统能源、农业后周期、必选消费
- 红线禁区（基本面长期向下）：房地产链、传统白酒、传统零售、旧软件外包

每个板块通过关键词匹配获得标签，支持行业板块和概念板块。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# 分类标签定义
# ======================================================================

@dataclass(frozen=True)
class MacroLabel:
    """宏观分类标签。"""

    category: str          # "tech" | "cycle" | "redline" | "neutral"
    sub_category: str      # 细分赛道
    display_name: str      # 中文显示名
    description: str       # 简要说明
    icon: str = ""         # 显示图标


# 科技主线六大赛道关键词
TECH_TRACKS: dict[str, dict] = {
    "chip": {
        "display": "芯片/半导体/算力",
        "icon": "🔧",
        "desc": "打底基建层 — 掘金热潮中的卖铲人",
        "keywords": [
            "半导体", "芯片", "集成电路", "算力", "光刻", "封装", "EDA",
            "GPU", "存储", "DRAM", "NAND", "晶圆", "刻蚀", "先进封装",
            "CPO", "HBM", "国产替代", "信创", "RISC-V",
        ],
    },
    "ai_model": {
        "display": "大模型/Agent应用",
        "icon": "🧠",
        "desc": "超级入口层 — 寻找下一个国民级应用",
        "keywords": [
            "人工智能", "大模型", "AIGC", "ChatGPT", "Agent", "大数据",
            "自然语言", "机器学习", "深度学习", "生成式AI", "智能算力",
            "数据要素", "云计算", "算力租赁", "AI应用",
        ],
    },
    "robot": {
        "display": "机器人/自动驾驶",
        "icon": "🤖",
        "desc": "物理世界落地 — AI脱虚向实，未来第一大支柱产业",
        "keywords": [
            "机器人", "人形机器人", "减速器", "伺服", "控制器",
            "自动驾驶", "智能驾驶", "智能汽车", "车联网", "激光雷达",
            "无人驾驶", "智能座舱", "线控底盘", "具身智能",
        ],
    },
    "ai_medical": {
        "display": "AI医疗",
        "icon": "🏥",
        "desc": "高精尖降维 — AI缩短新药研发周期",
        "keywords": [
            "创新药", "CRO", "生物医药", "基因", "细胞治疗",
            "AI制药", "医疗AI", "精准医疗", "基因编辑",
            "ADC", "mRNA", "GLP-1",
        ],
    },
    "space": {
        "display": "商业航天",
        "icon": "🚀",
        "desc": "高精尖降维 — 低轨卫星与太空基建加速",
        "keywords": [
            "商业航天", "卫星", "火箭", "航天", "低轨",
            "卫星互联网", "北斗", "遥感", "空天",
        ],
    },
    "new_energy_vehicle": {
        "display": "新能源车/智能电动",
        "icon": "🚗",
        "desc": "汽车是长了轮子的机器人，智能电动大趋势",
        "keywords": [
            "新能源车", "新能源汽车", "锂电池", "动力电池", "充电桩",
            "固态电池", "钠离子电池", "电池回收", "换电",
        ],
    },
}

# 周期主线五段论关键词
CYCLE_STAGES: dict[str, dict] = {
    "precious_metal": {
        "display": "贵金属",
        "stage": 1,
        "icon": "🥇",
        "desc": "阶段一：乱世避险，美元信用衰退",
        "status": "高位区",
        "keywords": [
            "黄金", "白银", "贵金属", "金矿",
        ],
    },
    "base_metal": {
        "display": "基本金属",
        "stage": 2,
        "icon": "🔩",
        "desc": "阶段二：新能源基建天量需求，铜铝供需错配",
        "status": "强共识区",
        "keywords": [
            "铜", "铝", "锌", "镍", "锡", "工业金属",
            "有色金属", "稀有金属", "稀土", "钨", "钴", "锂",
            "小金属",
        ],
    },
    "energy": {
        "display": "传统能源",
        "stage": 3,
        "icon": "🔥",
        "desc": "阶段三：地缘博弈，边打边撤",
        "status": "临界点",
        "keywords": [
            "煤炭", "石油", "天然气", "油气", "页岩",
            "石油开采", "油服", "OPEC",
        ],
    },
    "agriculture": {
        "display": "农业后周期",
        "stage": 4,
        "icon": "🌾",
        "desc": "阶段四：能源传导至农资，周期洼地重点布局",
        "status": "重点布局区",
        "keywords": [
            "农业", "化肥", "农药", "种子", "转基因", "种植",
            "饲料", "农资", "生猪", "养殖", "农机",
        ],
    },
    "consumer_staples": {
        "display": "必选消费",
        "stage": 5,
        "icon": "🛒",
        "desc": "阶段五：CPI温和传导，估值历史底部",
        "status": "底部左侧潜伏区",
        "keywords": [
            "食品饮料", "乳业", "调味品", "日化", "家居用品",
            "零食", "方便食品", "食品加工", "必选消费",
        ],
    },
}

# 红线禁区关键词
REDLINE_ZONES: dict[str, dict] = {
    "real_estate": {
        "display": "房地产及泛地产链",
        "icon": "🏚️",
        "desc": "人口结构与城镇化见顶，周期无法逆转",
        "keywords": [
            "房地产", "地产", "物业", "家装", "装修", "建材",
            "水泥", "钢铁", "建筑装饰", "房屋建设",
        ],
    },
    "liquor": {
        "display": "传统白酒",
        "icon": "🍶",
        "desc": "商务宴请需求塌方，年轻消费断代",
        "keywords": [
            "白酒", "酿酒", "啤酒",
        ],
    },
    "old_retail": {
        "display": "传统零售/低端餐饮",
        "icon": "🏪",
        "desc": "居民预期收入下降与消费降级双重挤压",
        "keywords": [
            "百货", "超市", "零售", "餐饮",
        ],
    },
    "old_software": {
        "display": "旧软件外包/客服",
        "icon": "💻",
        "desc": "将遭遇Agent的无情清洗",
        "keywords": [
            "软件外包", "IT服务", "呼叫中心", "客服外包",
            "系统集成", "传统软件",
        ],
    },
}


class SectorClassifier:
    """板块宏观分类器。

    根据板块名称进行关键词匹配，为每个板块打上宏观分类标签。
    支持自定义关键词扩展。
    """

    def __init__(
        self,
        tech_tracks: dict | None = None,
        cycle_stages: dict | None = None,
        redline_zones: dict | None = None,
    ) -> None:
        self.tech_tracks = tech_tracks or TECH_TRACKS
        self.cycle_stages = cycle_stages or CYCLE_STAGES
        self.redline_zones = redline_zones or REDLINE_ZONES
        # 构建扁平化查找表
        self._lookup = self._build_lookup()

    def _build_lookup(self) -> list[tuple[str, MacroLabel]]:
        """构建关键词 → 标签的查找表。"""
        lookup: list[tuple[str, MacroLabel]] = []

        for key, track in self.tech_tracks.items():
            label = MacroLabel(
                category="tech",
                sub_category=key,
                display_name=track["display"],
                description=track["desc"],
                icon=track["icon"],
            )
            for kw in track["keywords"]:
                lookup.append((kw, label))

        for key, stage in self.cycle_stages.items():
            label = MacroLabel(
                category="cycle",
                sub_category=key,
                display_name=stage["display"],
                description=stage["desc"],
                icon=stage["icon"],
            )
            for kw in stage["keywords"]:
                lookup.append((kw, label))

        for key, zone in self.redline_zones.items():
            label = MacroLabel(
                category="redline",
                sub_category=key,
                display_name=zone["display"],
                description=zone["desc"],
                icon=zone["icon"],
            )
            for kw in zone["keywords"]:
                lookup.append((kw, label))

        return lookup

    def classify(self, board_name: str) -> MacroLabel:
        """对单个板块名称进行分类。

        Parameters
        ----------
        board_name : str
            板块名称，如 "半导体"、"房地产开发"。

        Returns
        -------
        MacroLabel
            匹配到的宏观标签。未匹配返回 neutral 标签。
        """
        for keyword, label in self._lookup:
            if keyword in board_name or board_name in keyword:
                return label

        return MacroLabel(
            category="neutral",
            sub_category="other",
            display_name="其他",
            description="未归类板块",
            icon="⚪",
        )

    def classify_dataframe(self, df: pd.DataFrame, name_col: str = "name") -> pd.DataFrame:
        """批量分类板块 DataFrame。

        Parameters
        ----------
        df : pd.DataFrame
            包含板块名称列的 DataFrame。
        name_col : str
            板块名称列名。

        Returns
        -------
        pd.DataFrame
            增加 macro_category, macro_sub, macro_display, macro_icon 列。
        """
        df = df.copy()

        categories = []
        subs = []
        displays = []
        icons = []
        descs = []

        for name in df[name_col]:
            label = self.classify(str(name))
            categories.append(label.category)
            subs.append(label.sub_category)
            displays.append(label.display_name)
            icons.append(label.icon)
            descs.append(label.description)

        df["macro_category"] = categories
        df["macro_sub"] = subs
        df["macro_display"] = displays
        df["macro_icon"] = icons
        df["macro_desc"] = descs

        return df

    def get_category_summary(self, df: pd.DataFrame) -> dict[str, dict]:
        """获取各分类的汇总统计。

        Parameters
        ----------
        df : pd.DataFrame
            已分类的 DataFrame（需包含 macro_category 列）。

        Returns
        -------
        dict
            各分类的板块数量、平均涨幅等统计。
        """
        if "macro_category" not in df.columns:
            df = self.classify_dataframe(df)

        summary = {}
        for cat in ["tech", "cycle", "redline", "neutral"]:
            subset = df[df["macro_category"] == cat]
            cat_names = {
                "tech": "🗡️ 科技主线",
                "cycle": "🛡️ 周期主线",
                "redline": "🚫 红线禁区",
                "neutral": "⚪ 其他",
            }
            summary[cat] = {
                "display_name": cat_names.get(cat, cat),
                "count": len(subset),
                "avg_change": (
                    subset["change_pct"].mean() if "change_pct" in subset.columns
                    and not subset.empty
                    else 0.0
                ),
                "boards": (
                    subset["name"].tolist() if "name" in subset.columns
                    else []
                ),
            }

        return summary

    @staticmethod
    def get_all_tech_keywords() -> list[str]:
        """返回所有科技赛道关键词。"""
        keywords = []
        for track in TECH_TRACKS.values():
            keywords.extend(track["keywords"])
        return keywords

    @staticmethod
    def get_all_cycle_keywords() -> list[str]:
        """返回所有周期赛道关键词。"""
        keywords = []
        for stage in CYCLE_STAGES.values():
            keywords.extend(stage["keywords"])
        return keywords

    @staticmethod
    def get_all_redline_keywords() -> list[str]:
        """返回所有红线禁区关键词。"""
        keywords = []
        for zone in REDLINE_ZONES.values():
            keywords.extend(zone["keywords"])
        return keywords
