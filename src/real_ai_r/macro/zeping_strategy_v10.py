"""泽平宏观 V10 — 三维度信号 × 产业链敏感性 × 卖铲人溢出

设计哲学（汲取 V1→V9.3 经验）：
  V5/V7/V8 走领域先验路线 (WF Sharpe 1.5→2.86)
  V9/V9.2/V9.3 走算法学习路线 (-3.27 → +2.10, 受 149 天数据量限制)
  → 数据不足时，领域知识是主导信号源。V10 回归先验，但用泽平方法论深化。

泽平宏观方法论的三根支柱，V10 全部吸收：

1. 三维度制度框架（替代 V8 单一 regime_score）
   - 经济周期: 周期板块动量 - 消费防御动量
   - 流动性: 全市场换手 + 主力净流入（代理利率）
   - 风险偏好: 上涨广度 + 龙头相对强度
   每一维独立 → 避免 V8 一维 min 抹杀信号

2. 产业链敏感性矩阵（替代 tech/cycle 二分）
   - 10 条产业链，每条有 (cycle, liq, risk) 三维敏感性系数
   - 链分 = 敏感性 · 三维信号
   - 板块最终 bonus = 所属链分之平均
   - 同属"科技"但 AI 算力/消费电子/半导体对三维度的反应截然不同

3. 卖铲人动态溢出（泽平招牌理论）
   - V5 的 SHOVEL_SELLER_TRACKS_V5 只是静态 +10 分
   - V10 让溢出动态化: 某链热起来 → 上游设备/原材料板块按链热度比例加分
   - "别押赛道，卖铲子" — 下游不确定时押上游增加赢率

继承 V8：
  - 红线过滤
  - 回撤制动（连续 N 日累计超额 < threshold → 制动所有 bonus）

参数精简（仅 6 个有经济含义的参数）:
  chain_bonus_max, shovel_factor, signal_lookback,
  drawdown_lookback, drawdown_threshold, chain_membership_boost
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from real_ai_r.macro.classifier import SectorClassifier, TECH_TRACKS_V5
from real_ai_r.macro.zeping_strategy import (
    ZepingBoardScore,
    ZepingMacroStrategy,
    ZepingParams,
    ZepingPredictionResult,
    ZepingWeights,
)
from real_ai_r.macro.zeping_strategy_v5 import ZepingMacroStrategyV5

logger = logging.getLogger(__name__)


# ======================================================================
# 产业链定义（泽平宏观产业视角）
# ======================================================================


@dataclass(frozen=True)
class V10Chain:
    """一条产业链的完整定义。

    语义:
        - all_keywords: 任一匹配 → 板块属于该链（获得 chain bonus）
        - upstream_keywords: 任一匹配 → 板块是该链上游（获得 shovel spillover）
        - sensitivity: (cycle, liquidity, risk) 三维敏感度 ∈ [-1, +1]
            正值表示该维度上升时链受益
    """

    name: str
    all_keywords: tuple[str, ...]
    upstream_keywords: tuple[str, ...]
    sensitivity: tuple[float, float, float]  # (cycle, liq, risk)


# 基于 TECH_TRACKS_V5 + 泽平经典分类扩展
# 系数参考《宏观经济学》周期/流动性/成长股对三因素的响应
DEFAULT_CHAINS: tuple[V10Chain, ...] = (
    # ---- 成长线 ----
    V10Chain(
        name="AI算力",
        all_keywords=(
            "半导体", "芯片", "算力", "光模块", "光通信",
            "服务器", "数据中心", "人工智能", "AI",
        ),
        upstream_keywords=(
            "半导体设备", "芯片设计", "晶圆", "光刻胶", "PCB", "CCL",
        ),
        sensitivity=(+0.2, +0.8, +1.0),
    ),
    V10Chain(
        name="新能源",
        all_keywords=(
            "锂电池", "光伏", "风电", "储能", "氢能", "充电桩",
            "新能源汽车", "电动车",
        ),
        upstream_keywords=(
            "锂矿", "正极材料", "负极材料", "电解液", "隔膜",
            "多晶硅", "硅料", "稀土", "钴",
        ),
        sensitivity=(+0.3, +0.7, +0.9),
    ),
    V10Chain(
        name="半导体",
        all_keywords=(
            "半导体", "芯片", "集成电路", "存储", "模拟芯片",
            "射频", "封装测试",
        ),
        upstream_keywords=(
            "半导体设备", "光刻机", "化学机械抛光", "湿电子化学品",
        ),
        sensitivity=(+0.4, +0.6, +0.8),
    ),
    V10Chain(
        name="消费电子",
        all_keywords=(
            "消费电子", "智能硬件", "手机", "笔记本", "可穿戴",
            "TWS", "VR", "AR", "显示器", "面板",
        ),
        upstream_keywords=("玻璃盖板", "摄像头模组", "触摸屏"),
        sensitivity=(+0.5, +0.3, +0.6),
    ),
    # ---- 弱周期/防御线 ----
    V10Chain(
        name="医药生物",
        all_keywords=(
            "医药", "生物", "医疗器械", "疫苗", "中药", "化药",
            "CXO", "创新药", "医疗服务",
        ),
        upstream_keywords=("原料药", "医药研发外包"),
        sensitivity=(-0.1, +0.4, +0.2),
    ),
    V10Chain(
        name="军工",
        all_keywords=(
            "军工", "航空", "航天", "船舶", "兵器", "核电",
            "国防",
        ),
        upstream_keywords=("军工材料", "钛合金", "碳纤维"),
        sensitivity=(+0.1, +0.2, +0.5),
    ),
    # ---- 强周期线 ----
    V10Chain(
        name="金融",
        all_keywords=(
            "银行", "保险", "证券", "多元金融", "信托", "券商",
        ),
        upstream_keywords=(),
        sensitivity=(+0.8, +0.6, -0.3),
    ),
    V10Chain(
        name="周期资源",
        all_keywords=(
            "煤炭", "石油", "天然气", "钢铁", "有色", "铜", "铝",
            "化工", "化学", "水泥", "玻璃", "造纸",
        ),
        upstream_keywords=("贵金属", "工业金属", "小金属"),
        sensitivity=(+1.0, +0.3, -0.2),
    ),
    # ---- 防御消费线 ----
    V10Chain(
        name="消费防御",
        all_keywords=(
            "食品", "饮料", "乳业", "调味品", "禽畜", "水产",
            "日化", "家电",
        ),
        upstream_keywords=("农业", "种植业", "禽畜养殖"),
        sensitivity=(-0.3, -0.1, -0.5),
    ),
    # ---- 新经济线 ----
    V10Chain(
        name="新经济",
        all_keywords=(
            "互联网", "传媒", "游戏", "影视", "广告", "教育",
            "电商", "云服务", "SaaS",
        ),
        upstream_keywords=("云计算", "IDC"),
        sensitivity=(+0.4, +0.5, +0.8),
    ),
)


# ======================================================================
# V10 参数（仅 6 个，全部有经济含义）
# ======================================================================


@dataclass(frozen=True)
class V10Params:
    """V10 策略参数。"""

    # ---- 产业链 bonus ----
    chain_bonus_max: float = 10.0       # 链敏感性分映射到 bonus 的最大值
    chain_membership_boost: float = 2.0  # 属于任一链的基础加分（非红线 + 可识别产业链）

    # ---- 卖铲人溢出 ----
    shovel_factor: float = 0.6          # 上游板块从所在链热度获得的比例

    # ---- 信号计算窗口 ----
    signal_lookback: int = 5            # 三维度信号回看天数

    # ---- 回撤制动（继承 V8 设计）----
    drawdown_lookback: int = 5
    drawdown_threshold: float = -8.0    # 最近 N 日累计超额 < 阈值 → 制动


# ======================================================================
# V10 主类
# ======================================================================


class ZepingMacroStrategyV10(ZepingMacroStrategy):
    """泽平宏观 V10 — 三维度 × 产业链 × 卖铲人 + V8 回撤制动。"""

    VERSION = "V10"

    def __init__(
        self,
        weights: ZepingWeights | None = None,
        params: ZepingParams | None = None,
        v10_params: V10Params | None = None,
        chains: tuple[V10Chain, ...] | None = None,
    ) -> None:
        super().__init__(weights, params)
        self._v5 = ZepingMacroStrategyV5(weights, params)
        self.classifier = SectorClassifier(tech_tracks=TECH_TRACKS_V5)
        self.v10 = v10_params or V10Params()
        self.chains = chains or DEFAULT_CHAINS

        # 状态
        self._excess_history: list[float] = []
        # 板块名 -> [匹配的链名...]（缓存，避免每次 predict 重新扫关键词）
        self._board_chain_cache: dict[str, list[str]] = {}
        self._board_upstream_cache: dict[str, list[str]] = {}
        # 板块 5 日动量近似（从 tech_history / 外部注入）
        self._cycle_momentum_history: list[float] = []
        self._defensive_momentum_history: list[float] = []

    # ------------------------------------------------------------------
    # 反馈接口（兼容 V8）
    # ------------------------------------------------------------------

    def record_excess(self, daily_excess: float) -> None:
        """记录日超额，触发回撤制动用。"""
        self._excess_history.append(float(daily_excess))

    def record_board_performance(
        self, board_name: str, excess: float,
    ) -> None:
        """V10 不需要此回调，保留为占位。"""
        return

    # ------------------------------------------------------------------
    # 主预测入口
    # ------------------------------------------------------------------

    def predict(
        self,
        board_df: pd.DataFrame | None = None,
        fund_df: pd.DataFrame | None = None,
        top_n: int = 10,
        tech_history: list[float] | None = None,
        cycle_history: list[float] | None = None,
    ) -> ZepingPredictionResult:
        """
        tech_history / cycle_history 在 V10 中分别表示:
          - 科技板块日均涨幅序列
          - 周期板块日均涨幅序列
        V10 用它们计算"经济周期维度"信号。
        """
        # Step 1: V5 基础评分（全量）
        v5_result = self._v5.predict(board_df=board_df, top_n=10_000)
        if not v5_result.predictions or board_df is None or len(board_df) == 0:
            return v5_result

        # Step 2: 三维度信号
        cycle_signal = self._compute_cycle_signal(
            tech_history, cycle_history, board_df,
        )
        liquidity_signal = self._compute_liquidity_signal(board_df, fund_df)
        risk_signal = self._compute_risk_preference_signal(board_df)
        signals = np.array(
            [cycle_signal, liquidity_signal, risk_signal], dtype=float,
        )

        # Step 3: 回撤制动检查
        in_brake = self._is_in_drawdown()

        # Step 4: 每条链的当期热度分 = sensitivity · signals
        chain_scores: dict[str, float] = {}
        for chain in self.chains:
            sens = np.asarray(chain.sensitivity, dtype=float)
            chain_scores[chain.name] = float(sens @ signals)

        # Step 5: 每板块 bonus 计算
        adjusted_scores: list[tuple[ZepingBoardScore, float]] = []
        for s in v5_result.predictions:
            if s.macro_category == "redline":
                continue  # 红线板块永远不选
            bonus = 0.0 if in_brake else self._compute_board_bonus(
                s.board_name, chain_scores,
            )
            final_score = s.total_score + bonus
            adjusted_scores.append((s, final_score))

        # Step 6: 排序取 top_n
        adjusted_scores.sort(key=lambda x: x[1], reverse=True)
        top_items = adjusted_scores[:top_n]

        # Step 7: 构造 summary
        regime_label = self._regime_label(cycle_signal, liquidity_signal, risk_signal)
        top_chains = sorted(
            chain_scores.items(), key=lambda x: -x[1],
        )[:3]
        summary_parts = [
            f"V10[{regime_label}]",
            f"cycle={cycle_signal:+.2f} liq={liquidity_signal:+.2f} risk={risk_signal:+.2f}",
            f"top_chains=" + ",".join(f"{k}({v:+.2f})" for k, v in top_chains),
        ]
        if in_brake:
            summary_parts.append("制动中")

        result_scores = [s for s, _ in top_items]
        return ZepingPredictionResult(
            predictions=result_scores,
            current_hot_stage=v5_result.current_hot_stage,
            current_hot_stage_name=v5_result.current_hot_stage_name,
            market_style=f"{v5_result.market_style} | {regime_label}",
            total_boards=v5_result.total_boards,
            filtered_redline=v5_result.filtered_redline,
            strategy_summary=" | ".join(summary_parts),
        )

    # ------------------------------------------------------------------
    # 三维度信号
    # ------------------------------------------------------------------

    def _compute_cycle_signal(
        self,
        tech_history: list[float] | None,
        cycle_history: list[float] | None,
        board_df: pd.DataFrame,
    ) -> float:
        """经济周期信号 ∈ [-1, +1]。

        优先级:
            1. 若外部提供 tech_history/cycle_history → 用 cycle - tech 5日均差
               (周期占优 → +signal；原理：经济上行周期板块占优)
            2. 否则从 board_df 内部估算 (周期板块均涨 - 科技板块均涨)
        """
        lookback = self.v10.signal_lookback
        if (
            tech_history is not None
            and cycle_history is not None
            and len(tech_history) >= lookback
            and len(cycle_history) >= lookback
        ):
            tech_avg = sum(tech_history[-lookback:]) / lookback
            cycle_avg = sum(cycle_history[-lookback:]) / lookback
            diff = cycle_avg - tech_avg  # 周期 - 科技 → 周期强 → +
            return float(np.tanh(diff / 2.0))  # 映射到 (-1, +1)

        # Fallback: 直接用当日截面
        cycle_names = {
            kw for c in self.chains
            if c.name in ("金融", "周期资源")
            for kw in c.all_keywords
        }
        tech_names = {
            kw for c in self.chains
            if c.name in ("AI算力", "半导体", "新经济")
            for kw in c.all_keywords
        }
        cycle_ret = self._mean_change_for_keywords(board_df, cycle_names)
        tech_ret = self._mean_change_for_keywords(board_df, tech_names)
        return float(np.tanh((cycle_ret - tech_ret) / 2.0))

    def _compute_liquidity_signal(
        self, board_df: pd.DataFrame, fund_df: pd.DataFrame | None,
    ) -> float:
        """流动性信号 ∈ [-1, +1]。

        来源:
            1. 全市场换手率分位（> 历史均值 → +；< 均值 → -）
            2. 主力净流入（fund_df 提供时）
        """
        turnover = pd.to_numeric(
            board_df.get("turnover_rate", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0.0)
        if len(turnover) == 0:
            turnover_signal = 0.0
        else:
            mean_to = float(turnover.mean())
            # 朴素: 2% 为换手中位数，高于 3% 偏热，低于 1.5% 偏冷
            if mean_to > 3.0:
                turnover_signal = 1.0
            elif mean_to > 2.0:
                turnover_signal = 0.5
            elif mean_to > 1.5:
                turnover_signal = 0.0
            elif mean_to > 1.0:
                turnover_signal = -0.5
            else:
                turnover_signal = -1.0

        # 主力净流入（如果 fund_df 可用且含总额字段）
        fund_signal = 0.0
        if fund_df is not None and len(fund_df) > 0:
            for col in ("main_net_inflow", "今日主力净流入-净额", "主力净额"):
                if col in fund_df.columns:
                    total = pd.to_numeric(
                        fund_df[col], errors="coerce",
                    ).fillna(0.0).sum()
                    # 总净流入 > 0 → +，单位亿
                    fund_signal = float(np.tanh(total / 100.0))
                    break

        return float(np.clip(
            0.6 * turnover_signal + 0.4 * fund_signal, -1.0, 1.0,
        ))

    def _compute_risk_preference_signal(
        self, board_df: pd.DataFrame,
    ) -> float:
        """风险偏好信号 ∈ [-1, +1]。

        来源:
            1. 上涨家数比（广度）: > 0.6 → 偏好高；< 0.4 → 偏好低
            2. 领涨股相对强度: 平均 lead_stock_pct（龙头涨得多说明情绪高）
        """
        rise = pd.to_numeric(
            board_df.get("rise_count", 0), errors="coerce",
        ).fillna(0.0)
        fall = pd.to_numeric(
            board_df.get("fall_count", 0), errors="coerce",
        ).fillna(0.0)
        denom = (rise + fall).sum()
        if denom < 1e-6:
            breadth_signal = 0.0
        else:
            breadth_ratio = float(rise.sum() / denom)
            # 0.5 中性；(ratio - 0.5) × 4 映射到 (-2, +2)，再 tanh
            breadth_signal = float(np.tanh((breadth_ratio - 0.5) * 4.0))

        lead_signal = 0.0
        if "lead_stock_pct" in board_df.columns:
            lead_mean = float(
                pd.to_numeric(
                    board_df["lead_stock_pct"], errors="coerce",
                ).fillna(0.0).mean(),
            )
            # 龙头平均涨 > 5% → 强情绪，< 0 → 弱
            lead_signal = float(np.tanh(lead_mean / 5.0))

        return float(np.clip(
            0.6 * breadth_signal + 0.4 * lead_signal, -1.0, 1.0,
        ))

    # ------------------------------------------------------------------
    # 板块 bonus 计算（产业链 + 卖铲人）
    # ------------------------------------------------------------------

    def _compute_board_bonus(
        self, board_name: str, chain_scores: dict[str, float],
    ) -> float:
        """计算单板块的 V10 调整分 = chain bonus + shovel bonus。"""
        matched_chains = self._match_chains(board_name)
        upstream_of = self._match_upstream(board_name)

        bonus = 0.0

        # 1. 产业链 bonus
        if matched_chains:
            # 平均链热度（避免一个板块同属多链时双倍加分）
            avg_score = sum(chain_scores[c] for c in matched_chains) / len(matched_chains)
            # 映射: avg_score ∈ [-3, +3] 约束到 [-bonus_max, +bonus_max]
            chain_bonus = float(
                np.clip(
                    avg_score * self.v10.chain_bonus_max / 2.5,
                    -self.v10.chain_bonus_max,
                    self.v10.chain_bonus_max,
                ),
            )
            bonus += chain_bonus + self.v10.chain_membership_boost

        # 2. 卖铲人溢出（只在所属链热度 > 0 时）
        if upstream_of:
            spillover = 0.0
            for chain_name in upstream_of:
                c_score = chain_scores.get(chain_name, 0.0)
                if c_score > 0:
                    spillover += c_score * self.v10.shovel_factor
            # 卖铲 spillover 上限: chain_bonus_max × 0.8
            bonus += float(np.clip(
                spillover, 0.0, self.v10.chain_bonus_max * 0.8,
            ))

        return bonus

    def _match_chains(self, board_name: str) -> list[str]:
        """返回板块所属的所有链名（缓存）。"""
        if board_name in self._board_chain_cache:
            return self._board_chain_cache[board_name]
        matched = [
            c.name for c in self.chains
            if any(kw in board_name for kw in c.all_keywords)
        ]
        self._board_chain_cache[board_name] = matched
        return matched

    def _match_upstream(self, board_name: str) -> list[str]:
        """返回板块作为上游的所有链名（卖铲人身份）。"""
        if board_name in self._board_upstream_cache:
            return self._board_upstream_cache[board_name]
        matched = [
            c.name for c in self.chains
            if c.upstream_keywords
            and any(kw in board_name for kw in c.upstream_keywords)
        ]
        self._board_upstream_cache[board_name] = matched
        return matched

    # ------------------------------------------------------------------
    # 回撤制动
    # ------------------------------------------------------------------

    def _is_in_drawdown(self) -> bool:
        lookback = self.v10.drawdown_lookback
        if len(self._excess_history) < lookback:
            return False
        recent_cum = sum(self._excess_history[-lookback:])
        return recent_cum < self.v10.drawdown_threshold

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_change_for_keywords(
        board_df: pd.DataFrame, keywords: set[str],
    ) -> float:
        if not keywords:
            return 0.0
        cp = pd.to_numeric(
            board_df.get("change_pct", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0.0)
        names = board_df["name"].astype(str).values
        mask = np.array(
            [any(kw in n for kw in keywords) for n in names],
            dtype=bool,
        )
        if mask.sum() == 0:
            return 0.0
        return float(cp.values[mask].mean())

    @staticmethod
    def _regime_label(cycle: float, liq: float, risk: float) -> str:
        """把三维度信号转为可读大环境标签（按泽平 4 象限）。"""
        total = cycle + liq + risk
        if risk > 0.3 and liq > 0.3:
            return "牛市普涨"
        if risk > 0.3 and cycle > 0.3:
            return "顺周期轮动"
        if risk < -0.3 and liq < -0.3:
            return "防御市"
        if abs(total) < 0.3:
            return "结构市"
        if risk < 0:
            return "避险"
        return "轮动"

    def reset(self) -> None:
        """重置状态（回测新窗口前用）。"""
        self._excess_history.clear()
        self._board_chain_cache.clear()
        self._board_upstream_cache.clear()


__all__ = [
    "ZepingMacroStrategyV10",
    "V10Params",
    "V10Chain",
    "DEFAULT_CHAINS",
]
