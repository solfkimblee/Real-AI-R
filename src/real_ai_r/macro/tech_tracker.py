"""科技六赛道追踪器

追踪康波第六波六大科技赛道的实时表现：
1. 芯片/半导体/算力 — 打底基建层
2. 大模型/Agent应用 — 超级入口层
3. 机器人/自动驾驶 — 物理世界落地
4. AI医疗 — 高精尖降维
5. 商业航天 — 高精尖降维
6. 新能源车/智能电动 — 智能电动大趋势

每个赛道通过匹配相关板块，汇总资金流、涨幅、活跃度等指标。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from real_ai_r.macro.classifier import TECH_TRACKS
from real_ai_r.sector.monitor import SectorMonitor

logger = logging.getLogger(__name__)


@dataclass
class TrackSnapshot:
    """单个科技赛道的实时快照。"""

    key: str                  # 赛道 key
    display: str              # 中文名
    icon: str                 # 图标
    description: str          # 简要说明
    avg_change_pct: float     # 平均涨跌幅
    total_rise: int           # 上涨家数合计
    total_fall: int           # 下跌家数合计
    avg_turnover: float       # 平均换手率
    top_lead_stock: str       # 领涨股票
    top_lead_pct: float       # 领涨股涨幅
    matched_boards: list[str] # 匹配到的板块列表
    heat_score: float         # 综合热度 0-100


class TechTracker:
    """科技六赛道追踪器。

    通过匹配板块关键词，实时追踪各科技赛道的表现。
    """

    def __init__(self, tech_tracks: dict | None = None) -> None:
        self.tracks = tech_tracks or TECH_TRACKS

    def track_all(self) -> list[TrackSnapshot]:
        """追踪所有科技赛道。

        Returns
        -------
        list[TrackSnapshot]
            各赛道的实时快照。
        """
        logger.info("开始追踪科技六赛道...")

        # 同时获取行业和概念板块以扩大匹配范围
        board_dfs = []
        for bt in ["industry", "concept"]:
            try:
                df = SectorMonitor.get_board_list(bt)
                df["_board_type"] = bt
                board_dfs.append(df)
            except Exception:
                logger.warning("获取%s板块数据失败", bt)

        if not board_dfs:
            return self._empty_tracks()

        all_boards = pd.concat(board_dfs, ignore_index=True)
        # 去重：同名板块保留概念板块优先
        all_boards = all_boards.drop_duplicates(subset=["name"], keep="last")

        results = []
        for key, track_info in self.tracks.items():
            snapshot = self._compute_track_snapshot(key, track_info, all_boards)
            results.append(snapshot)

        # 按热度排序
        results.sort(key=lambda t: t.heat_score, reverse=True)
        return results

    def get_top_tracks(self, top_n: int = 3) -> list[TrackSnapshot]:
        """获取最热的 N 个赛道。"""
        tracks = self.track_all()
        return tracks[:top_n]

    def get_track_comparison(self) -> pd.DataFrame:
        """获取各赛道对比 DataFrame。"""
        tracks = self.track_all()
        rows = []
        for t in tracks:
            rows.append({
                "赛道": f"{t.icon} {t.display}",
                "热度": t.heat_score,
                "平均涨幅%": t.avg_change_pct,
                "上涨/下跌": f"{t.total_rise}/{t.total_fall}",
                "换手率%": t.avg_turnover,
                "领涨股": t.top_lead_stock,
                "领涨涨幅%": t.top_lead_pct,
                "匹配板块数": len(t.matched_boards),
            })
        return pd.DataFrame(rows)

    def _compute_track_snapshot(
        self,
        key: str,
        track_info: dict,
        all_boards: pd.DataFrame,
    ) -> TrackSnapshot:
        """计算单个赛道的快照。"""
        keywords = track_info["keywords"]

        # 匹配板块
        matched = all_boards[
            all_boards["name"].apply(lambda n: any(kw in str(n) for kw in keywords))
        ]

        if matched.empty:
            return TrackSnapshot(
                key=key,
                display=track_info["display"],
                icon=track_info["icon"],
                description=track_info["desc"],
                avg_change_pct=0.0,
                total_rise=0,
                total_fall=0,
                avg_turnover=0.0,
                top_lead_stock="-",
                top_lead_pct=0.0,
                matched_boards=[],
                heat_score=0.0,
            )

        avg_change = matched["change_pct"].mean()
        avg_turnover = (
            matched["turnover_rate"].mean()
            if "turnover_rate" in matched.columns
            else 0.0
        )

        total_rise = int(
            matched["rise_count"].sum()
            if "rise_count" in matched.columns
            else 0
        )
        total_fall = int(
            matched["fall_count"].sum()
            if "fall_count" in matched.columns
            else 0
        )

        # 领涨股
        if "lead_stock_pct" in matched.columns and not matched.empty:
            top_idx = matched["lead_stock_pct"].idxmax()
            top_lead_stock = str(matched.loc[top_idx, "lead_stock"])
            top_lead_pct = float(matched.loc[top_idx, "lead_stock_pct"])
        else:
            top_lead_stock = "-"
            top_lead_pct = 0.0

        # 综合热度计算
        # 涨幅得分 (40%)
        momentum = max(0.0, min(100.0, 50.0 + avg_change * 10.0))
        # 活跃度得分 (30%)
        activity = min(avg_turnover * 10, 100.0)
        # 广度得分 (30%)
        total = total_rise + total_fall
        breadth = (total_rise / total * 100) if total > 0 else 50.0

        heat_score = round(
            0.40 * momentum + 0.30 * activity + 0.30 * breadth, 1,
        )

        return TrackSnapshot(
            key=key,
            display=track_info["display"],
            icon=track_info["icon"],
            description=track_info["desc"],
            avg_change_pct=round(avg_change, 2),
            total_rise=total_rise,
            total_fall=total_fall,
            avg_turnover=round(avg_turnover, 2),
            top_lead_stock=top_lead_stock,
            top_lead_pct=round(top_lead_pct, 2),
            matched_boards=matched["name"].tolist(),
            heat_score=heat_score,
        )

    def _empty_tracks(self) -> list[TrackSnapshot]:
        """数据获取失败时返回空快照。"""
        results = []
        for key, info in self.tracks.items():
            results.append(TrackSnapshot(
                key=key,
                display=info["display"],
                icon=info["icon"],
                description=info["desc"],
                avg_change_pct=0.0,
                total_rise=0,
                total_fall=0,
                avg_turnover=0.0,
                top_lead_stock="-",
                top_lead_pct=0.0,
                matched_boards=[],
                heat_score=0.0,
            ))
        return results
