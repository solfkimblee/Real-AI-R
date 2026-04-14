"""攻防组合推荐 — 科技矛 + 周期盾对冲组合

基于泽平宏观框架的核心阵型思路：
- 矛（进攻）：芯片/算力/机器人等科技真龙头，博取康波超额收益
- 盾（防御）：农资/种子/食品等周期洼地，防御通胀与地缘黑天鹅

组合原则：
1. 科技矛选择热度最高的赛道内领涨个股
2. 周期盾选择处于洼地（阶段四/五）的板块内低估值个股
3. 自动排除红线禁区标的
4. 动态调整攻防比例（默认 6:4）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from real_ai_r.macro.cycle_tracker import CycleTracker
from real_ai_r.macro.red_filter import RedLineFilter
from real_ai_r.macro.tech_tracker import TechTracker
from real_ai_r.sector.recommender import StockRecommender

logger = logging.getLogger(__name__)


@dataclass
class PortfolioSlot:
    """组合中的一个持仓槽位。"""

    role: str             # "attack" | "defense"
    role_display: str     # 中文角色名
    track: str            # 所属赛道/阶段
    board_name: str       # 板块名称
    stock_code: str       # 股票代码
    stock_name: str       # 股票名称
    price: float          # 最新价
    change_pct: float     # 涨跌幅
    score: float          # 评分
    reason: str           # 入选理由


@dataclass
class PortfolioResult:
    """攻防组合结果。"""

    attack_slots: list[PortfolioSlot]   # 进攻仓位
    defense_slots: list[PortfolioSlot]  # 防御仓位
    attack_ratio: float                  # 进攻比例
    defense_ratio: float                 # 防御比例
    summary: dict                        # 组合摘要


class AttackDefensePortfolio:
    """攻防组合构建器。

    综合科技赛道追踪、周期轮动追踪、红线过滤，
    自动构建科技矛 + 周期盾的对冲投资组合。
    """

    def __init__(
        self,
        attack_ratio: float = 0.6,
        attack_slots: int = 5,
        defense_slots: int = 4,
    ) -> None:
        """
        Parameters
        ----------
        attack_ratio : float
            进攻仓位比例 (0-1)，防御 = 1 - attack_ratio。
        attack_slots : int
            进攻持仓个股数量。
        defense_slots : int
            防御持仓个股数量。
        """
        self.attack_ratio = attack_ratio
        self.defense_ratio = 1.0 - attack_ratio
        self.n_attack = attack_slots
        self.n_defense = defense_slots
        self.tech_tracker = TechTracker()
        self.cycle_tracker = CycleTracker()
        self.red_filter = RedLineFilter()

    def build(self) -> PortfolioResult:
        """构建攻防组合。

        Returns
        -------
        PortfolioResult
            包含进攻和防御仓位的完整组合。
        """
        logger.info("开始构建攻防组合（攻:守 = %.0f:%.0f）...",
                     self.attack_ratio * 100, self.defense_ratio * 100)

        attack_slots = self._build_attack()
        defense_slots = self._build_defense()

        summary = {
            "attack_count": len(attack_slots),
            "defense_count": len(defense_slots),
            "attack_ratio": self.attack_ratio,
            "defense_ratio": self.defense_ratio,
            "attack_tracks": list({s.track for s in attack_slots}),
            "defense_tracks": list({s.track for s in defense_slots}),
        }

        return PortfolioResult(
            attack_slots=attack_slots,
            defense_slots=defense_slots,
            attack_ratio=self.attack_ratio,
            defense_ratio=self.defense_ratio,
            summary=summary,
        )

    def _build_attack(self) -> list[PortfolioSlot]:
        """构建进攻仓位 — 从最热科技赛道中选股。"""
        logger.info("构建科技矛（进攻仓位）...")
        slots: list[PortfolioSlot] = []

        try:
            tracks = self.tech_tracker.track_all()
        except Exception:
            logger.warning("科技赛道追踪失败")
            return slots

        # 按热度排序，选前3个赛道
        top_tracks = sorted(tracks, key=lambda t: t.heat_score, reverse=True)[:3]

        stocks_per_track = max(1, self.n_attack // len(top_tracks)) if top_tracks else 0
        remainder = self.n_attack - stocks_per_track * len(top_tracks)

        for i, track in enumerate(top_tracks):
            if not track.matched_boards:
                continue

            n = stocks_per_track + (1 if i < remainder else 0)
            board_name = track.matched_boards[0]

            try:
                recommended = StockRecommender.recommend(
                    board_name=board_name,
                    board_type="industry",
                    top_n=n,
                    sort_by="composite",
                )
            except Exception:
                # 行业板块失败，尝试概念板块
                try:
                    recommended = StockRecommender.recommend(
                        board_name=board_name,
                        board_type="concept",
                        top_n=n,
                        sort_by="composite",
                    )
                except Exception:
                    logger.warning("赛道 [%s] 选股失败", track.display)
                    continue

            if recommended.empty:
                continue

            # 过滤红线
            if "name" in recommended.columns:
                recommended = self.red_filter.filter_stocks(recommended)

            for _, stock in recommended.head(n).iterrows():
                slots.append(PortfolioSlot(
                    role="attack",
                    role_display="🗡️ 科技矛",
                    track=f"{track.icon} {track.display}",
                    board_name=board_name,
                    stock_code=str(stock.get("code", "")),
                    stock_name=str(stock.get("name", "")),
                    price=float(stock.get("price", 0)),
                    change_pct=float(stock.get("change_pct", 0)),
                    score=float(stock.get("composite_score", 0)),
                    reason=(
                        f"科技赛道热度{track.heat_score}分, "
                        f"综合评分{stock.get('composite_score', 0):.1f}"
                    ),
                ))

            if len(slots) >= self.n_attack:
                break

        return slots[:self.n_attack]

    def _build_defense(self) -> list[PortfolioSlot]:
        """构建防御仓位 — 从周期洼地（阶段四/五）中选股。"""
        logger.info("构建周期盾（防御仓位）...")
        slots: list[PortfolioSlot] = []

        try:
            stages = self.cycle_tracker.track()
        except Exception:
            logger.warning("周期追踪失败")
            return slots

        # 优先选阶段四（农业）和阶段五（必选消费）
        defense_stages = [s for s in stages if s.stage in (4, 5)]
        # 如果阶段四/五没有匹配到板块，也考虑其他低温阶段
        if not defense_stages:
            defense_stages = sorted(stages, key=lambda s: s.temperature)[:2]

        stocks_per_stage = max(1, self.n_defense // len(defense_stages)) if defense_stages else 0
        remainder = self.n_defense - stocks_per_stage * len(defense_stages)

        for i, stage in enumerate(defense_stages):
            if not stage.matched_boards:
                continue

            n = stocks_per_stage + (1 if i < remainder else 0)
            board_name = stage.matched_boards[0]

            try:
                recommended = StockRecommender.recommend(
                    board_name=board_name,
                    board_type="industry",
                    top_n=n * 2,  # 多选一些以备过滤
                    sort_by="composite",
                )
            except Exception:
                try:
                    recommended = StockRecommender.recommend(
                        board_name=board_name,
                        board_type="concept",
                        top_n=n * 2,
                        sort_by="composite",
                    )
                except Exception:
                    logger.warning("阶段 [%s] 选股失败", stage.display)
                    continue

            if recommended.empty:
                continue

            # 过滤红线
            if "name" in recommended.columns:
                recommended = self.red_filter.filter_stocks(recommended)

            for _, stock in recommended.head(n).iterrows():
                slots.append(PortfolioSlot(
                    role="defense",
                    role_display="🛡️ 周期盾",
                    track=f"{stage.icon} {stage.display}",
                    board_name=board_name,
                    stock_code=str(stock.get("code", "")),
                    stock_name=str(stock.get("name", "")),
                    price=float(stock.get("price", 0)),
                    change_pct=float(stock.get("change_pct", 0)),
                    score=float(stock.get("composite_score", 0)),
                    reason=(
                        f"周期阶段{stage.stage}({stage.framework_status}), "
                        f"温度{stage.temperature}"
                    ),
                ))

            if len(slots) >= self.n_defense:
                break

        return slots[:self.n_defense]

    def to_dataframe(self, result: PortfolioResult) -> pd.DataFrame:
        """将组合结果转换为 DataFrame。"""
        rows = []
        for slot in result.attack_slots + result.defense_slots:
            rows.append({
                "角色": slot.role_display,
                "赛道": slot.track,
                "板块": slot.board_name,
                "代码": slot.stock_code,
                "名称": slot.stock_name,
                "最新价": slot.price,
                "涨跌幅%": slot.change_pct,
                "评分": slot.score,
                "入选理由": slot.reason,
            })
        return pd.DataFrame(rows)
