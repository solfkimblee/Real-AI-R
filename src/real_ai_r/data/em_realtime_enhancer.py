"""EM实时增强器 — 利用496板块实时数据对V1.1输出做二次微调

设计理念：
    EM的496个细分板块实时快照可用，包含当日最新的资金流向、换手率、
    上涨/下跌家数等信息。这些信息比THS 90个大类板块更细粒度。

    增强逻辑：
    1. V1.1输出Top10板块（基于THS 90板块历史数据）
    2. 将Top10映射到EM 496个细分板块
    3. 计算三个实时增强因子：资金攻击强度、上涨集中度、龙头溢价
    4. 对Top10排序做微调（权重10%）

    集成公式：
    final_score = v1_score * 0.90 + realtime_score * 0.10

使用方法：
    enhancer = EMRealtimeEnhancer()
    enhanced_result = enhancer.enhance(v11_result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# 增强参数
# ======================================================================

@dataclass(frozen=True)
class EMEnhancerParams:
    """EM实时增强器参数。"""

    v1_weight: float = 0.90              # V1原始评分权重
    realtime_weight: float = 0.10        # 实时因子权重

    # 三大实时因子的内部权重（归一化到1.0）
    attack_intensity_weight: float = 0.40  # 资金攻击强度
    rise_concentration_weight: float = 0.35  # 上涨集中度
    lead_premium_weight: float = 0.25      # 龙头溢价


# ======================================================================
# 实时因子结果
# ======================================================================

@dataclass
class RealtimeFactors:
    """单个板块的实时增强因子。"""

    board_name: str
    em_board_name: str = ""              # EM中匹配的板块名

    attack_intensity: float = 0.0        # 资金攻击强度
    rise_concentration: float = 0.0      # 上涨集中度
    lead_premium: float = 0.0           # 龙头溢价

    realtime_score: float = 0.0          # 综合实时得分 (0-100)
    enhanced_score: float = 0.0          # 最终增强得分

    matched: bool = False                # 是否在EM中找到匹配


@dataclass
class EnhancementResult:
    """增强结果汇总。"""

    factors: list[RealtimeFactors]
    em_board_count: int = 0              # EM板块总数
    matched_count: int = 0               # 成功匹配数
    enhancement_applied: bool = False    # 是否应用了增强
    description: str = ""


# ======================================================================
# EM实时增强器
# ======================================================================

class EMRealtimeEnhancer:
    """EM实时增强器 — 对V1.1输出做二次微调。

    使用方法：
        enhancer = EMRealtimeEnhancer()
        # 获取EM实时数据
        em_df = enhancer.fetch_em_realtime()
        # 增强V1.1输出
        enhanced = enhancer.enhance(v11_predictions, em_df)
    """

    def __init__(
        self,
        params: EMEnhancerParams | None = None,
    ) -> None:
        self.params = params or EMEnhancerParams()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def enhance(
        self,
        predictions: list,
        em_df: pd.DataFrame | None = None,
    ) -> EnhancementResult:
        """对V1.1输出的预测列表做实时增强。

        Parameters
        ----------
        predictions : list[ZepingBoardScore]
            V1.1输出的预测列表。
        em_df : pd.DataFrame | None
            EM实时数据。如果为 None，自动获取。

        Returns
        -------
        EnhancementResult
            增强结果。
        """
        if em_df is None:
            em_df = self.fetch_em_realtime()

        if em_df is None or em_df.empty:
            return EnhancementResult(
                factors=[],
                enhancement_applied=False,
                description="EM实时数据获取失败，跳过增强",
            )

        # Step 1: 计算全市场统计量（用于归一化）
        market_stats = self._compute_market_stats(em_df)

        # Step 2: 为每个预测板块计算实时因子
        factors: list[RealtimeFactors] = []
        matched_count = 0

        for pred in predictions:
            factor = self._compute_factors_for_board(
                pred, em_df, market_stats,
            )
            factors.append(factor)
            if factor.matched:
                matched_count += 1

        # Step 3: 对预测列表排序做微调
        if matched_count > 0:
            self._apply_reorder(predictions, factors)

        return EnhancementResult(
            factors=factors,
            em_board_count=len(em_df),
            matched_count=matched_count,
            enhancement_applied=matched_count > 0,
            description=(
                f"EM增强: {matched_count}/{len(predictions)}板块匹配成功, "
                f"EM总板块{len(em_df)}个"
            ),
        )

    # ------------------------------------------------------------------
    # EM实时数据获取
    # ------------------------------------------------------------------

    @staticmethod
    def fetch_em_realtime(board_type: str = "industry") -> pd.DataFrame | None:
        """获取EM实时板块数据。"""
        import akshare as ak

        try:
            if board_type == "industry":
                df = ak.stock_board_industry_name_em()
            else:
                df = ak.stock_board_concept_name_em()

            col_map = {
                "排名": "rank", "板块名称": "name", "板块代码": "code",
                "最新价": "close", "涨跌额": "change_amount",
                "涨跌幅": "change_pct", "总市值": "total_market_cap",
                "换手率": "turnover_rate", "上涨家数": "rise_count",
                "下跌家数": "fall_count", "领涨股票": "lead_stock",
                "领涨股票-涨跌幅": "lead_stock_change",
            }
            df = df.rename(columns=col_map)
            return df

        except Exception as e:
            logger.warning("EM实时数据获取失败: %s", e)
            return None

    # ------------------------------------------------------------------
    # 实时因子计算
    # ------------------------------------------------------------------

    def _compute_market_stats(
        self,
        em_df: pd.DataFrame,
    ) -> dict:
        """计算全市场统计量（用于因子归一化）。"""
        change_pct = pd.to_numeric(em_df.get("change_pct", pd.Series()), errors="coerce")
        turnover = pd.to_numeric(em_df.get("turnover_rate", pd.Series()), errors="coerce")
        lead_change = pd.to_numeric(em_df.get("lead_stock_change", pd.Series()), errors="coerce")

        return {
            "avg_change": float(change_pct.mean()) if not change_pct.empty else 0.0,
            "std_change": float(change_pct.std()) if not change_pct.empty else 1.0,
            "avg_turnover": float(turnover.mean()) if not turnover.empty else 0.0,
            "avg_lead_change": float(lead_change.mean()) if not lead_change.empty else 0.0,
            "std_lead_change": float(lead_change.std()) if not lead_change.empty else 1.0,
        }

    def _compute_factors_for_board(
        self,
        pred,
        em_df: pd.DataFrame,
        market_stats: dict,
    ) -> RealtimeFactors:
        """为单个板块计算实时增强因子。"""
        board_name = pred.board_name

        # 模糊匹配EM板块
        em_row = self._fuzzy_match_board(board_name, em_df)

        if em_row is None:
            return RealtimeFactors(
                board_name=board_name,
                matched=False,
            )

        em_name = str(em_row.get("name", ""))
        change_pct = _safe_float(em_row.get("change_pct"))
        turnover = _safe_float(em_row.get("turnover_rate"))
        rise_count = _safe_int(em_row.get("rise_count"))
        fall_count = _safe_int(em_row.get("fall_count"))
        lead_change = _safe_float(em_row.get("lead_stock_change"))

        # ---- 因子1: 资金攻击强度 ----
        # (领涨股涨幅 - 板块平均涨幅) * 换手率
        # 高值 = 龙头带动效应强
        attack_raw = (lead_change - change_pct) * max(turnover, 0.1)
        attack_norm = self._normalize(
            attack_raw,
            center=market_stats["avg_lead_change"] * market_stats["avg_turnover"],
            scale=max(market_stats["std_lead_change"] * market_stats["avg_turnover"], 0.1),
        )

        # ---- 因子2: 上涨集中度 ----
        # 上涨家数 / 总家数
        total_stocks = rise_count + fall_count
        rise_ratio = rise_count / total_stocks if total_stocks > 0 else 0.5
        rise_norm = self._normalize(rise_ratio, center=0.5, scale=0.3)

        # ---- 因子3: 龙头溢价 ----
        # 领涨股涨幅 / 板块涨幅 (>1表示龙头效应强)
        if abs(change_pct) > 0.01:
            lead_premium_raw = lead_change / change_pct
        else:
            lead_premium_raw = 1.0
        lead_norm = self._normalize(lead_premium_raw, center=1.5, scale=1.0)

        # ---- 综合实时得分 ----
        p = self.params
        realtime_score = (
            p.attack_intensity_weight * attack_norm
            + p.rise_concentration_weight * rise_norm
            + p.lead_premium_weight * lead_norm
        ) * 100  # 缩放到0-100

        # ---- 最终增强得分 ----
        original_score = pred.total_score
        enhanced_score = (
            p.v1_weight * original_score
            + p.realtime_weight * realtime_score
        )

        return RealtimeFactors(
            board_name=board_name,
            em_board_name=em_name,
            attack_intensity=round(attack_norm, 3),
            rise_concentration=round(rise_norm, 3),
            lead_premium=round(lead_norm, 3),
            realtime_score=round(realtime_score, 2),
            enhanced_score=round(enhanced_score, 2),
            matched=True,
        )

    def _apply_reorder(
        self,
        predictions: list,
        factors: list[RealtimeFactors],
    ) -> None:
        """根据增强得分微调预测排序。"""
        # 建立映射
        score_map = {f.board_name: f.enhanced_score for f in factors if f.matched}

        # 对已匹配的板块按增强得分重排
        for pred in predictions:
            if pred.board_name in score_map:
                pred.total_score = score_map[pred.board_name]
                pred.reasons.append("📡EM实时增强")

        predictions.sort(key=lambda s: s.total_score, reverse=True)

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _fuzzy_match_board(
        ths_name: str,
        em_df: pd.DataFrame,
    ) -> pd.Series | None:
        """模糊匹配THS板块名到EM板块。

        匹配策略：
        1. 精确匹配
        2. 包含匹配（THS名包含在EM名中，或反之）
        3. 关键词匹配（取前2个字符匹配）
        """
        names = em_df["name"].astype(str)

        # 精确匹配
        exact = em_df[names == ths_name]
        if not exact.empty:
            return exact.iloc[0]

        # 包含匹配
        contains = em_df[names.str.contains(ths_name, na=False, regex=False)]
        if not contains.empty:
            return contains.iloc[0]

        reverse = em_df[names.apply(lambda n: n in ths_name)]
        if not reverse.empty:
            return reverse.iloc[0]

        # 前缀匹配（至少2个字符）
        if len(ths_name) >= 2:
            prefix = ths_name[:2]
            prefix_match = em_df[names.str.startswith(prefix, na=False)]
            if len(prefix_match) == 1:
                return prefix_match.iloc[0]

        return None

    @staticmethod
    def _normalize(value: float, center: float, scale: float) -> float:
        """将值归一化到0-1范围（sigmoid-like）。"""
        if scale <= 0:
            return 0.5
        z = (value - center) / scale
        # 简单截断归一化
        return max(0.0, min(1.0, 0.5 + z * 0.25))


# ======================================================================
# 工具函数
# ======================================================================

def _safe_float(val) -> float:
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _safe_int(val) -> int:
    if val is None:
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0
