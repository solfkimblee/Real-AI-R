"""泽平宏观 V14 — V13 + 最新直播方法论信号（波浪位置/共识度/通胀传导/赛道优先级/逆向埋伏）。

V14 = V13（LightGBM + 全泽平联动特征）+ 6大新维度：

基于2026年4月泽平宏观最新直播纪要，新增以下可量化信号：

D. V14新增特征（~16个）：
   11. 波浪位置: wave_position, wave_momentum, wave_maturity, wave_drawdown_risk
       — "24年9月起点，现在3→4浪过渡，未来3-5月还有大调整"
   12. 共识度/拥挤度逆向: board_crowding, crowd_reversion, consensus_divergence,
       contrarian_score — "市场一旦形成共识，反着来"
   13. 通胀传导预警: agri_consumer_heat, inflation_stage, music_ending_signal
       — "农产品+消费品涨→通胀来了→音乐结束"
   14. 科技赛道优先级: track_priority_weight, priority_adjusted_heat
       — "大模型>算力>机器人>自动驾驶>AI医药>商业航天"
   15. 逆向猪周期: pig_contrarian, agri_momentum_direction
       — "买在猪价跌时，卖在猪价涨时"
   16. 红线扩充: is_redline_v14
       — 新增中药、低端芯片禁区

总特征: V13的~56个 + V14的~16个 = ~72个特征

训练方式：固定训练（同V13），不做滚动再训练。
Horizon：单1天horizon（同V13），不做多horizon融合。

接口:
    drop-in V13/V11 兼容:
    - fit(panel_df) — 初始训练
    - predict(board_df, fund_df=None, top_n=10, tech_history=None, cycle_history=None)
    - record_excess(daily_excess)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore[assignment]


# ======================================================================
# 导入V13的全部基础设施
# ======================================================================

from real_ai_r.macro.zeping_strategy_v13_lgbm import (
    # 分类体系
    TECH_CHAIN_KEYWORDS,
    CYCLE_CHAIN_KEYWORDS,
    CYCLE_STAGE_ORDER,
    DEFENSIVE_KEYWORDS,
    UPSTREAM_KEYWORDS,
    CHAIN_SENSITIVITY,
    # 分类函数
    _classify_board,
    _get_chain_id,
    _get_cycle_stage,
    _is_redline,
    _is_upstream,
    _get_tech_track,
    # V13特征构造函数
    _build_ts_features,
    _build_linkage_features,
    _build_cycle_stage_features,
    _build_regime_features,
    _build_track_heat_features,
    _build_upstream_features,
    _build_redline_features,
    _build_market_structure_features,
    _apply_cs_rank,
    # 推理用
    build_inference_features as v13_build_inference_features,
    # 预测结果容器
    ZepingPrediction,
    ZepingPredictionResult,
)


# ======================================================================
# V14 新增常量
# ======================================================================

# 科技赛道优先级权重（基于泽平最新排序: 大模型>算力>机器人>自动驾驶>AI医药>商业航天）
TECH_TRACK_PRIORITY: dict[str, float] = {
    "ai_software": 1.00,      # 大模型 — "下一个腾讯"
    "chip": 0.90,             # 算力 — "还有100倍空间"
    "robot_auto": 0.75,       # 机器人+自动驾驶 — "爆发前夜"+"大规模落地"
    "medical": 0.60,          # AI医药 — "永恒主题"
    "space_military": 0.50,   # 商业航天 — "发射量几十倍增长"
    "new_energy": 0.55,       # 新能源 — "固态电池/储能坚定看好"
    "telecom": 0.45,          # 通信 — "光纤光模块看好"
    "none": 0.0,              # 非科技板块
}

# V14红线扩充关键词（在V13基础上新增中药、低端芯片）
REDLINE_KEYWORDS_V14: list[str] = [
    # V13原有
    "房地产", "地产", "物业", "家装", "装修", "水泥",
    "百货", "超市", "零售", "餐饮",
    "软件外包", "IT服务", "系统集成",
    # V14新增
    "中药", "中成药", "中医",      # 泽平明确"不看了"
    "低端芯片",                    # "未来1-2年过剩，没有门槛"
]

# 农业板块关键词（用于逆向猪周期信号）
AGRI_KEYWORDS: list[str] = [
    "农业", "化肥", "农药", "种子", "饲料", "生猪", "养殖",
]

# 消费品关键词（用于通胀传导预警）
CONSUMER_KEYWORDS: list[str] = [
    "食品饮料", "乳业", "调味品", "日化", "零食", "方便食品", "食品加工",
]


# ======================================================================
# V14 新增特征构造函数
# ======================================================================


def _build_wave_features(panel: pd.DataFrame) -> pd.DataFrame:
    """构造波浪位置特征（V14新增）。

    基于泽平波浪理论: "24年9月起点，现在3→4浪，未来3-5月还有大调整"
    通过市场整体技术指标估算当前波浪位置:
    - 用市场均值的多尺度动量交叉判断波浪阶段
    - 用加速度/减速度判断波浪成熟度
    """
    panel = panel.copy()

    # 计算市场日均涨幅
    mkt_daily = panel.groupby("date")["change_pct"].mean().sort_index()
    mkt_daily_df = mkt_daily.reset_index()
    mkt_daily_df.columns = ["date", "mkt_avg_ret"]

    # 多尺度市场动量
    mkt_daily_df["mkt_cum_20d"] = mkt_daily_df["mkt_avg_ret"].rolling(
        20, min_periods=1
    ).sum()
    mkt_daily_df["mkt_cum_60d"] = mkt_daily_df["mkt_avg_ret"].rolling(
        60, min_periods=1
    ).sum()
    mkt_daily_df["mkt_cum_120d"] = mkt_daily_df["mkt_avg_ret"].rolling(
        120, min_periods=1
    ).sum()

    # 波浪位置: 短期动量 vs 长期动量的关系
    # 短>长>0 = 上涨浪加速, 短>0>长 = 上涨浪减速, 0>短>长 = 下跌浪加速
    mkt_daily_df["wave_position"] = np.where(
        mkt_daily_df["mkt_cum_20d"] > 0,
        np.where(
            mkt_daily_df["mkt_cum_60d"] > 0,
            2.0,   # 双正: 强上涨浪
            1.0,   # 短正长负: 反弹浪
        ),
        np.where(
            mkt_daily_df["mkt_cum_60d"] > 0,
            -1.0,  # 短负长正: 调整浪
            -2.0,  # 双负: 下跌浪
        ),
    )

    # 波浪动量: 短期动量变化率（加速 vs 减速）
    mkt_daily_df["wave_momentum"] = (
        mkt_daily_df["mkt_cum_20d"]
        - mkt_daily_df["mkt_cum_20d"].shift(5).fillna(0.0)
    )

    # 波浪成熟度: 累计涨幅 / 历史最大累计涨幅（接近1=接近顶部）
    mkt_daily_df["_cum_total"] = mkt_daily_df["mkt_avg_ret"].cumsum()
    mkt_daily_df["_cum_max"] = mkt_daily_df["_cum_total"].expanding().max()
    mkt_daily_df["wave_maturity"] = np.where(
        mkt_daily_df["_cum_max"].abs() > 1e-6,
        mkt_daily_df["_cum_total"] / (mkt_daily_df["_cum_max"] + 1e-6),
        0.5,
    )
    mkt_daily_df["wave_maturity"] = mkt_daily_df["wave_maturity"].clip(0.0, 1.0)

    # 回撤风险: 距前高的回撤幅度
    mkt_daily_df["wave_drawdown_risk"] = (
        mkt_daily_df["_cum_max"] - mkt_daily_df["_cum_total"]
    )

    # 合并到panel
    merge_cols = ["date", "wave_position", "wave_momentum",
                  "wave_maturity", "wave_drawdown_risk"]
    panel = panel.merge(
        mkt_daily_df[merge_cols], on="date", how="left",
    )
    for col in merge_cols[1:]:
        panel[col] = panel[col].fillna(0.0)

    return panel


def _build_crowding_features(panel: pd.DataFrame) -> pd.DataFrame:
    """构造共识度/拥挤度逆向特征（V14新增）。

    泽平核心交易哲学:
    - "市场一旦形成共识，反着来"
    - "买在无人问津时，卖在人声鼎沸处"
    - "绝望中重生，争议中上涨，狂欢中崩盘"
    """
    panel = panel.copy()

    # --- 板块拥挤度: 高换手+高动量 = 拥挤 ---
    if "turnover_rate" in panel.columns and "ret_5d" in panel.columns:
        # 每日板块换手率排名 + 动量排名的平均值
        panel["_to_rank"] = panel.groupby("date")["turnover_rate"].transform(
            lambda x: x.rank(pct=True)
        )
        panel["_mom_rank"] = panel.groupby("date")["ret_5d"].transform(
            lambda x: x.rank(pct=True)
        )
        panel["board_crowding"] = (panel["_to_rank"] + panel["_mom_rank"]) / 2.0
        panel = panel.drop(columns=["_to_rank", "_mom_rank"], errors="ignore")
    else:
        panel["board_crowding"] = 0.5

    # --- 拥挤度均值回归信号: 板块5日涨幅的z-score取反 ---
    if "ret_5d" in panel.columns:
        panel["_ret5_mean"] = panel.groupby("date")["ret_5d"].transform("mean")
        panel["_ret5_std"] = panel.groupby("date")["ret_5d"].transform("std")
        panel["crowd_reversion"] = -(
            (panel["ret_5d"] - panel["_ret5_mean"])
            / (panel["_ret5_std"] + 1e-6)
        ).clip(-3, 3)
        panel = panel.drop(columns=["_ret5_mean", "_ret5_std"], errors="ignore")
    else:
        panel["crowd_reversion"] = 0.0

    # --- 共识分歧: 短期动量方向 vs 中期动量方向不一致 ---
    if "ret_3d" in panel.columns and "ret_10d" in panel.columns:
        # 短多中空 = 可能见顶; 短空中多 = 可能见底
        panel["consensus_divergence"] = np.sign(panel["ret_3d"]) - np.sign(panel["ret_10d"])
    else:
        panel["consensus_divergence"] = 0.0

    # --- 逆向综合分: 融合拥挤度和均值回归 ---
    panel["contrarian_score"] = (
        (1.0 - panel["board_crowding"]) * 0.5
        + panel["crowd_reversion"] / 6.0 * 0.3  # 归一化到0~1
        + panel["consensus_divergence"] / 4.0 * 0.2
    )

    return panel


def _build_inflation_features(panel: pd.DataFrame) -> pd.DataFrame:
    """构造通胀传导预警特征（V14新增）。

    泽平核心判断: "农产品+消费品涨→通胀来了→货币收紧→音乐结束"
    大宗五段论传导: 贵金属→基本金属→能源→农产品→消费品
    当传导到农产品/消费品阶段，意味着通胀即将兑现，牛市接近尾声。
    """
    panel = panel.copy()

    # --- 农业+消费品综合热度 ---
    def _is_agri(name: str) -> bool:
        return any(kw in name for kw in AGRI_KEYWORDS)

    def _is_consumer(name: str) -> bool:
        return any(kw in name for kw in CONSUMER_KEYWORDS)

    panel["_is_agri"] = panel["name"].apply(_is_agri).astype(float)
    panel["_is_consumer"] = panel["name"].apply(_is_consumer).astype(float)

    # 每日农业和消费品的平均涨幅
    for label, col_name in [("_is_agri", "agri_avg"), ("_is_consumer", "consumer_avg")]:
        daily_avg = (
            panel[panel[label] > 0]
            .groupby("date")["change_pct"]
            .mean()
        )
        panel = panel.merge(
            daily_avg.rename(col_name).reset_index(),
            on="date",
            how="left",
        )
        panel[col_name] = panel[col_name].fillna(0.0)

    # 农业+消费品联合热度 (两者同时涨 = 通胀传导信号)
    panel["agri_consumer_heat"] = panel["agri_avg"] + panel["consumer_avg"]

    # --- 通胀传导阶段: 五段论热度重心位置 ---
    if "_sector_type" not in panel.columns:
        panel["_sector_type"] = panel["name"].apply(_classify_board)

    # 用stage_heat_1~5判断当前热度重心（如果已有的话）
    stage_cols = [c for c in panel.columns if c.startswith("stage_heat_")]
    if len(stage_cols) >= 5:
        # 加权重心: 越靠后=越接近通胀
        def _inflation_stage(row: pd.Series) -> float:
            heats = [row.get(f"stage_heat_{i}", 0.0) for i in range(1, 6)]
            total = sum(heats) + 1e-9
            weighted = sum(h * i for i, h in enumerate(heats, 1))
            return weighted / total
        panel["inflation_stage"] = panel.apply(_inflation_stage, axis=1)
    else:
        # 简化: 用农业+消费品占总体涨幅的比重
        daily_total = panel.groupby("date")["change_pct"].mean()
        panel = panel.merge(
            daily_total.rename("_mkt_total").reset_index(),
            on="date",
            how="left",
        )
        panel["inflation_stage"] = (
            panel["agri_consumer_heat"] / (panel["_mkt_total"].abs() + 1e-6)
        ).clip(-5, 5)
        panel = panel.drop(columns=["_mkt_total"], errors="ignore")

    # --- "音乐结束"信号: 农产品+消费品同时强 + 总市场高位 ---
    # 20日滚动的农业消费品热度
    date_heat = (
        panel.drop_duplicates("date")[["date", "agri_consumer_heat"]]
        .sort_values("date")
        .set_index("date")
    )
    rolled_heat = date_heat["agri_consumer_heat"].rolling(20, min_periods=1).mean()
    panel = panel.merge(
        rolled_heat.rename("music_ending_signal").reset_index(),
        on="date",
        how="left",
    )
    panel["music_ending_signal"] = panel["music_ending_signal"].fillna(0.0)

    # 清理中间列
    panel = panel.drop(
        columns=["_is_agri", "_is_consumer", "agri_avg", "consumer_avg"],
        errors="ignore",
    )

    return panel


def _build_track_priority_features(panel: pd.DataFrame) -> pd.DataFrame:
    """构造科技赛道优先级特征（V14新增）。

    泽平明确排序: 大模型>算力>机器人>自动驾驶>AI医药>商业航天
    高优先级赛道获得更高的权重加分。
    """
    panel = panel.copy()

    if "_tech_track" not in panel.columns:
        panel["_tech_track"] = panel["name"].apply(_get_tech_track)

    panel["track_priority_weight"] = panel["_tech_track"].map(
        TECH_TRACK_PRIORITY
    ).fillna(0.0)

    # 优先级调整后的赛道热度
    if "track_heat" in panel.columns:
        panel["priority_adjusted_heat"] = (
            panel["track_heat"] * (0.5 + 0.5 * panel["track_priority_weight"])
        )
    else:
        panel["priority_adjusted_heat"] = panel["track_priority_weight"] * 50.0

    return panel


def _build_pig_cycle_features(panel: pd.DataFrame) -> pd.DataFrame:
    """构造逆向猪周期特征（V14新增）。

    泽平核心观点: "猪价跌→买入农业板块, 猪价涨→卖出"
    与大多数人相反: "大部分人非要等猪价涨了才进去"

    通过农业板块的动量方向作为逆向信号:
    - 农业板块下跌中 → pig_contrarian > 0 (买入信号)
    - 农业板块上涨中 → pig_contrarian < 0 (卖出信号)
    """
    panel = panel.copy()

    if "_is_agri" not in panel.columns:
        panel["_is_agri"] = panel["name"].apply(
            lambda n: 1.0 if any(kw in n for kw in AGRI_KEYWORDS) else 0.0
        )

    # 每日农业板块平均涨幅
    agri_daily = (
        panel[panel["_is_agri"] > 0]
        .groupby("date")["change_pct"]
        .mean()
    )

    # 20日滚动农业动量（正=涨势, 负=跌势）
    agri_daily_sorted = agri_daily.sort_index()
    agri_mom_20d = agri_daily_sorted.rolling(20, min_periods=1).sum()

    panel = panel.merge(
        agri_mom_20d.rename("_agri_mom_20d").reset_index(),
        on="date",
        how="left",
    )
    panel["_agri_mom_20d"] = panel["_agri_mom_20d"].fillna(0.0)

    # 逆向猪周期信号: 取反（农业跌→正信号, 农业涨→负信号）
    panel["pig_contrarian"] = -panel["_agri_mom_20d"]

    # 农业动量方向（正=上涨趋势, 负=下跌趋势, 用于其他板块参考）
    agri_mom_5d = agri_daily_sorted.rolling(5, min_periods=1).sum()
    panel = panel.merge(
        agri_mom_5d.rename("_agri_mom_5d").reset_index(),
        on="date",
        how="left",
    )
    panel["agri_momentum_direction"] = np.sign(
        panel["_agri_mom_5d"].fillna(0.0)
    )

    # 清理中间列
    panel = panel.drop(
        columns=["_is_agri", "_agri_mom_20d", "_agri_mom_5d"],
        errors="ignore",
    )

    return panel


def _build_redline_v14_features(panel: pd.DataFrame) -> pd.DataFrame:
    """构造V14扩充红线特征。

    在V13红线基础上新增: 中药("中药","中成药","中医"), 低端芯片。
    """
    panel = panel.copy()
    panel["is_redline_v14"] = panel["name"].apply(
        lambda n: 1.0 if any(kw in n for kw in REDLINE_KEYWORDS_V14) else 0.0
    )
    return panel


# ======================================================================
# V14 完整数据构造
# ======================================================================


def build_training_data_v14(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """构造V14完整训练数据（V13全部特征 + V14新增6类特征 + 目标）。

    Returns:
        (featured_panel, feature_columns)
    """
    # ===== V13全部基础特征 =====
    # V11基础特征
    panel = _build_ts_features(panel)
    # V12原有联动特征（12个）
    panel = _build_linkage_features(panel)
    # V13: 五段论轮动
    panel = _build_cycle_stage_features(panel)
    # V13: 三维度制度信号
    panel = _build_regime_features(panel)
    # V13: 赛道热度
    panel = _build_track_heat_features(panel)
    # V13: 卖铲人/上下游
    panel = _build_upstream_features(panel)
    # V13: 红线风险（保留原始的）
    panel = _build_redline_features(panel)
    # V13: 市场分化
    panel = _build_market_structure_features(panel)

    # ===== V14 新增特征 =====
    # 11. 波浪位置（4个特征）
    panel = _build_wave_features(panel)
    # 12. 共识度/拥挤度逆向（4个特征）
    panel = _build_crowding_features(panel)
    # 13. 通胀传导预警（3个特征）
    panel = _build_inflation_features(panel)
    # 14. 科技赛道优先级（2个特征）
    panel = _build_track_priority_features(panel)
    # 15. 逆向猪周期（2个特征）
    panel = _build_pig_cycle_features(panel)
    # 16. V14扩充红线（1个特征）
    panel = _build_redline_v14_features(panel)

    # ===== 目标变量 =====
    daily_avg = panel.groupby("date")["change_pct"].transform("mean")
    panel["excess"] = panel["change_pct"] - daily_avg

    grouped = panel.groupby("name", group_keys=False)
    panel["target_ret"] = grouped["excess"].shift(-1)
    panel["target_rank"] = panel.groupby("date")["target_ret"].transform(
        lambda x: x.rank()
    )

    # ===== 截面排名归一化 =====
    panel, feat_cols = _apply_cs_rank(panel)

    # 清理临时列
    panel = panel.drop(columns=["_sector_type", "_chain_id", "_tech_track"],
                       errors="ignore")

    return panel, feat_cols


def build_inference_features_v14(
    board_df: pd.DataFrame,
    history: list[pd.DataFrame],
) -> tuple[pd.DataFrame, list[str]]:
    """构造V14推理特征（预测时用）。"""
    if not history:
        return pd.DataFrame(), []

    # 用V13的推理基础设施构造V13特征
    panel_for_today, v13_feat_cols = v13_build_inference_features(
        board_df=board_df,
        history=history,
    )

    if panel_for_today.empty:
        return panel_for_today, []

    # 在V13推理结果基础上, 追加V14特征
    # 需要先拿到含历史的完整panel来计算V14特征
    # 重要: 必须先跑V13的全部特征工程管线, 否则V14 builders缺少
    # 依赖的中间特征(ret_5d, ret_3d, turnover_rate, track_heat等)
    all_frames = list(history) + [board_df.copy()]
    combined = pd.concat(all_frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])

    # 先运行V13的全部特征工程管线（V14 builders依赖这些中间特征）
    combined = _build_ts_features(combined)
    combined = _build_linkage_features(combined)
    combined = _build_cycle_stage_features(combined)
    combined = _build_regime_features(combined)
    combined = _build_track_heat_features(combined)
    combined = _build_upstream_features(combined)
    combined = _build_redline_features(combined)
    combined = _build_market_structure_features(combined)

    # 计算V14特征（现在combined上有完整的V13中间特征）
    combined = _build_wave_features(combined)
    combined = _build_crowding_features(combined)
    combined = _build_inflation_features(combined)
    combined = _build_track_priority_features(combined)
    combined = _build_pig_cycle_features(combined)
    combined = _build_redline_v14_features(combined)

    # 只取最新一天
    today_date = combined["date"].max()
    today_rows = combined[combined["date"] == today_date].copy()

    # 截面排名归一化（只做V14新增特征）
    v14_raw_features = [
        "wave_position", "wave_momentum", "wave_maturity", "wave_drawdown_risk",
        "board_crowding", "crowd_reversion", "consensus_divergence", "contrarian_score",
        "agri_consumer_heat", "inflation_stage", "music_ending_signal",
        "track_priority_weight", "priority_adjusted_heat",
        "pig_contrarian", "agri_momentum_direction",
        "is_redline_v14",
    ]
    v14_feat_cols = []
    for col in v14_raw_features:
        if col in today_rows.columns:
            rank_col = f"{col}_cs_rank"
            today_rows[rank_col] = today_rows[col].rank(pct=True)
            v14_feat_cols.append(rank_col)

    # 合并V13和V14特征到panel_for_today
    # 先用name来join V14特征
    v14_cols_to_merge = ["name"] + v14_feat_cols
    v14_cols_available = [c for c in v14_cols_to_merge if c in today_rows.columns]

    if len(v14_cols_available) > 1:
        panel_for_today = panel_for_today.merge(
            today_rows[v14_cols_available].drop_duplicates("name"),
            on="name",
            how="left",
        )

    all_feat_cols = v13_feat_cols + v14_feat_cols
    return panel_for_today, all_feat_cols


# ======================================================================
# V14 策略类
# ======================================================================


class ZepingLGBMStrategyV14:
    """V14 LightGBM + 全泽平方法论联动特征 + 最新直播信号。

    V14 = V13（56特征）+ 波浪位置 + 共识度逆向 + 通胀传导 + 赛道优先级 + 猪周期逆向 + 红线扩充
    固定训练，单1天horizon，无滚动再训练。

    用法:
        v14 = ZepingLGBMStrategyV14()
        v14.fit(panel_train_df)
        result = v14.predict(board_df, top_n=10)
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_history_days: int = 30,
        lgbm_params: dict[str, Any] | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_history_days = max_history_days
        self.lgbm_params = lgbm_params or {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 100,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
            "verbosity": -1,
            "n_jobs": -1,
        }
        self.model: lgb.LGBMRegressor | None = None
        self.feature_cols: list[str] = []
        self._history: list[pd.DataFrame] = []
        self._excess_history: list[float] = []
        self._fitted = False

    def fit(self, panel_df: pd.DataFrame) -> None:
        """在训练面板上训练 LightGBM 模型。"""
        if lgb is None:
            raise ImportError("lightgbm not installed")

        print(f"[V14] 构造截面排名特征 + 泽平联动 + 直播新信号... (panel: {panel_df.shape})")
        featured, feat_cols = build_training_data_v14(panel_df)

        # 去掉没有 target 的行（最后一天）
        featured = featured.dropna(subset=["target_rank"])
        self.feature_cols = feat_cols

        X = featured[self.feature_cols].fillna(0.5)
        y = featured["target_rank"]

        print(f"[V14] 训练 LightGBM... (samples: {len(X)}, features: {len(self.feature_cols)})")
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            **self.lgbm_params,
        )
        self.model.fit(X, y)
        self._fitted = True

        # 打印特征重要性 Top15
        importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_cols,
        ).sort_values(ascending=False)
        print("[V14] Top15 重要特征:")
        for feat, imp in importance.head(15).items():
            print(f"       {feat}: {imp}")
        print(f"[V14] 总特征数: {len(self.feature_cols)}")

        # 保存最后 max_history_days 天数据作为初始历史
        panel_df = panel_df.copy()
        panel_df["date"] = pd.to_datetime(panel_df["date"])
        dates = sorted(panel_df["date"].unique())
        for d in dates[-self.max_history_days:]:
            self._history.append(
                panel_df[panel_df["date"] == d].reset_index(drop=True)
            )

    def predict(
        self,
        board_df: pd.DataFrame | None = None,
        fund_df: pd.DataFrame | None = None,
        top_n: int = 10,
        tech_history: list[float] | None = None,
        cycle_history: list[float] | None = None,
    ) -> ZepingPredictionResult:
        """预测当日 Top-N 板块。"""
        if board_df is None or board_df.empty:
            return ZepingPredictionResult(predictions=[])

        if not self._fitted or self.model is None:
            return ZepingPredictionResult(predictions=[])

        # 构造今日特征
        today_featured, infer_feat_cols = build_inference_features_v14(
            board_df=board_df,
            history=self._history[-self.max_history_days:],
        )

        if today_featured.empty:
            self._history.append(board_df.copy())
            return ZepingPredictionResult(predictions=[])

        # 对齐特征列
        for col in self.feature_cols:
            if col not in today_featured.columns:
                today_featured[col] = 0.5

        X = today_featured[self.feature_cols].fillna(0.5)

        # 预测截面排名
        scores = self.model.predict(X)
        today_featured = today_featured.copy()
        today_featured["pred_score"] = scores

        # 排序选 Top-N
        top_boards = today_featured.nlargest(top_n, "pred_score")[["name", "pred_score"]]

        predictions = [
            ZepingPrediction(board_name=row["name"], score=row["pred_score"])
            for _, row in top_boards.iterrows()
        ]

        # 记录今日数据到历史
        self._history.append(board_df.copy())
        if len(self._history) > self.max_history_days:
            self._history = self._history[-self.max_history_days:]

        return ZepingPredictionResult(predictions=predictions)

    def record_excess(self, daily_excess: float) -> None:
        """记录每日超额收益。"""
        self._excess_history.append(daily_excess)

    def save_model(self, path: str | Path) -> None:
        """保存模型到文件。"""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_cols": self.feature_cols,
                "lgbm_params": self.lgbm_params,
                "n_estimators": self.n_estimators,
            }, f)

    def load_model(self, path: str | Path) -> None:
        """从文件加载模型。"""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_cols = data["feature_cols"]
        self._fitted = True
