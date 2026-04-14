"""投资决策检查清单模块

基于泽平投资方法论，提供结构化的投资决策辅助工具：
- 五问决策框架
- 宏观/赛道/龙头/催化剂综合评估
- 决策建议生成
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ChecklistItem:
    """检查清单单项。"""

    question: str
    description: str
    answer: str = ""
    score: int = 0  # 0-100
    weight: float = 0.2


@dataclass
class InvestmentChecklist:
    """投资决策检查清单。"""

    target_name: str = ""
    target_type: str = "stock"  # "stock" | "sector" | "track"
    items: list[ChecklistItem] = field(default_factory=list)
    overall_score: float = 0.0
    recommendation: str = ""
    risk_notes: list[str] = field(default_factory=list)

    def compute_overall_score(self) -> float:
        """计算综合评分。"""
        if not self.items:
            return 0.0
        total_weight = sum(item.weight for item in self.items)
        if total_weight == 0:
            return 0.0
        self.overall_score = sum(
            item.score * item.weight for item in self.items
        ) / total_weight
        return self.overall_score

    def generate_recommendation(self) -> str:
        """根据评分生成投资建议。"""
        score = self.compute_overall_score()

        if score >= 80:
            self.recommendation = "强烈推荐 — 符合方法论核心标准，可重仓布局"
        elif score >= 65:
            self.recommendation = "推荐关注 — 基本面良好，等待催化剂确认后加仓"
        elif score >= 50:
            self.recommendation = "观望为主 — 部分条件不满足，建议小仓位试探"
        elif score >= 35:
            self.recommendation = "谨慎参与 — 风险较大，不建议重仓"
        else:
            self.recommendation = "回避 — 不符合投资框架，建议远离"

        return self.recommendation


def create_zepin_checklist() -> list[ChecklistItem]:
    """创建泽平方法论五问决策清单模板。

    基于方法论核心框架：
    1. 是否未来1-3年主线？
    2. 趋势还是周期？
    3. 有真实订单和业绩？
    4. 产业链最硬环节？
    5. 追高还是低位布局？
    6. 是否触碰红线？
    """
    return [
        ChecklistItem(
            question="这是不是未来1-3年的主线？",
            description=(
                "判断标的是否处于长期景气度上行的主赛道。"
                "科技主线（AI/芯片/机器人）= 高分，"
                "周期主线（贵金属/农业）= 中分，"
                "传统夕阳行业 = 低分"
            ),
            weight=0.25,
        ),
        ChecklistItem(
            question="这是趋势机会，还是周期轮动？",
            description=(
                "趋势资产（AI科技）可穿越周期，适合长持；"
                "周期资产（大宗商品）需严格波段操作。"
                "清楚自己持有的是趋势还是周期，决定持仓策略。"
            ),
            weight=0.15,
        ),
        ChecklistItem(
            question="这家公司有没有真实订单和业绩兑现？",
            description=(
                "有核心技术壁垒 + 真实订单/收入 = 高分；"
                "纯概念炒作、没有业绩兑现 = 低分。"
                "关注：100万台量产交付或B端大合同等商业闭环。"
            ),
            weight=0.25,
        ),
        ChecklistItem(
            question="它是不是产业链里最硬的环节？",
            description=(
                "优先选择产业链关键环节的\"卖铲人\"。"
                "算力链 > 应用层，核心零部件 > 终端组装。"
                "绑定头部客户或平台的供应商优先。"
            ),
            weight=0.20,
        ),
        ChecklistItem(
            question="当前是在追高，还是等待催化剂前布局？",
            description=(
                "最佳买点：调整充分 + 催化剂即将兑现（如财报季、政策发布）。"
                "高位追涨 = 低分，低位左侧布局 = 高分。"
                "参考：RSI > 70 过热，RSI < 30 超卖。"
            ),
            weight=0.15,
        ),
    ]


def create_redline_checklist() -> list[ChecklistItem]:
    """创建红线避雷检查清单。"""
    return [
        ChecklistItem(
            question="是否属于宏观红线禁区？",
            description=(
                "房地产及泛地产链、传统白酒（除茅台级超高端）、"
                "传统零售实体与低端餐饮、被AI替代的旧软件外包/客服 "
                "— 这些行业基本面长期向下，应坚决回避。"
            ),
            weight=1.0,
        ),
    ]


def evaluate_with_macro_context(
    target_name: str,
    is_tech: bool = False,
    is_cycle: bool = False,
    is_redline: bool = False,
    cycle_stage: int = 0,
) -> InvestmentChecklist:
    """结合宏观状态自动评估标的。

    Parameters
    ----------
    target_name : str
        标的名称（板块或个股）。
    is_tech : bool
        是否属于科技主线。
    is_cycle : bool
        是否属于周期主线。
    is_redline : bool
        是否属于红线禁区。
    cycle_stage : int
        周期阶段（0=非周期, 1-5）。

    Returns
    -------
    InvestmentChecklist
        自动填充的检查清单。
    """
    checklist = InvestmentChecklist(
        target_name=target_name,
        items=create_zepin_checklist(),
    )

    # 自动评分：问题1 — 是否主线
    if is_redline:
        checklist.items[0].score = 10
        checklist.items[0].answer = "红线禁区，非主线方向"
        checklist.risk_notes.append("触碰宏观红线禁区，建议回避")
    elif is_tech:
        checklist.items[0].score = 90
        checklist.items[0].answer = "科技主线，属于1-3年长期景气赛道"
    elif is_cycle:
        checklist.items[0].score = 60
        checklist.items[0].answer = "周期主线，需关注轮动节奏"
    else:
        checklist.items[0].score = 40
        checklist.items[0].answer = "非核心主线"

    # 自动评分：问题2 — 趋势 vs 周期
    if is_tech:
        checklist.items[1].score = 85
        checklist.items[1].answer = "趋势资产，可穿越周期"
    elif is_cycle:
        if cycle_stage in (4, 5):
            checklist.items[1].score = 70
            checklist.items[1].answer = f"周期资产（阶段{cycle_stage}），当前处于布局窗口"
        elif cycle_stage == 3:
            checklist.items[1].score = 50
            checklist.items[1].answer = "周期资产（能源阶段），边打边撤"
        else:
            checklist.items[1].score = 55
            checklist.items[1].answer = f"周期资产（阶段{cycle_stage}）"
    else:
        checklist.items[1].score = 30
        checklist.items[1].answer = "既非趋势也非主力周期品种"

    # 问题3-5需要用户手动输入或更多数据

    checklist.compute_overall_score()
    checklist.generate_recommendation()
    return checklist
