"""泽平宏观 V13 — LightGBM + 全泽平方法论联动特征。

V13 = V11（截面因子LightGBM）+ 全面泽平宏观方法论特征扩展。

基于消融实验结论：V12c（仅联动特征）是唯一有正面贡献的改进方向。
V13在V12c的12个联动特征基础上，深度挖掘泽平方法论的6大维度：

特征体系（~20个基础特征 + ~20个泽平联动特征 = ~40个特征）：

A. V11基础特征（20个）：
   - 多尺度动量/波动（3d/5d/10d/20d）
   - 换手率因子（5d/20d均值, ratio）
   - 广度因子, 领涨因子

B. V12原有联动特征（12个）：
   1. 产业链联动: tech/cycle/defensive_chain_avg
   2. 科技vs周期跷跷板: tc_spread_3d/5d/10d
   3. 同赛道共振: chain_peer_avg/std, chain_relative
   4. 交叉特征: mom_turnover_cross, vol_breadth_cross, mom_divergence

C. V13新增特征（~20个）：
   5. 五段论轮动: stage_heat_1~5, hot_stage_id, stage_distance, stage_lead_lag
   6. 三维度制度信号: cycle_signal, liquidity_signal, risk_pref_signal,
      regime_score, board_sensitivity_score
   7. 赛道热度: track_heat, track_rank, track_heat_ma5, track_relative_heat
   8. 卖铲人/上下游: is_upstream, downstream_heat, shovel_premium
   9. 红线风险: is_redline
   10. 市场分化: market_concentration, tech_cycle_divergence, chain_dispersion

训练方式：固定训练（同V11），不做滚动再训练。
Horizon：单1天horizon（同V11），不做多horizon融合。

接口:
    drop-in V11 兼容:
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
# 泽平宏观板块分类体系
# ======================================================================

# 科技产业链关键词（基于classifier.py的TECH_TRACKS_V5）
TECH_CHAIN_KEYWORDS: dict[str, list[str]] = {
    "chip": ["半导体", "芯片", "集成电路", "算力", "光刻", "封装",
             "分立器件", "印制电路板", "面板", "电子化学品", "光学光电子"],
    "ai_software": ["人工智能", "大模型", "AIGC", "大数据", "云计算",
                    "软件开发", "通用软件", "应用软件", "计算机", "互联网",
                    "游戏", "传媒"],
    "robot_auto": ["机器人", "自动驾驶", "智能驾驶", "智能汽车",
                   "自动化设备", "工控", "激光设备", "仪器仪表"],
    "medical": ["医疗器械", "医疗服务", "创新药", "生物医药", "生物制品",
                "化学制药", "体外诊断", "疫苗"],
    "new_energy": ["光伏", "风电", "锂电池", "充电桩", "新能源车",
                   "新能源汽车", "储能", "逆变器"],
    "telecom": ["通信", "电信运营商", "消费电子"],
    "space_military": ["航天", "卫星", "航空装备", "国防军工", "军工电子"],
}

# 周期产业链关键词 — 大宗商品五段论（基于classifier.py的CYCLE_STAGES）
CYCLE_CHAIN_KEYWORDS: dict[str, list[str]] = {
    "precious_metal": ["黄金", "白银", "贵金属"],
    "base_metal": ["有色金属", "稀有金属", "稀土", "铜", "铝", "锌",
                   "小金属"],
    "energy": ["煤炭", "石油", "天然气", "油气"],
    "agriculture": ["农业", "化肥", "农药", "种子", "饲料", "生猪", "养殖"],
    "consumer_staples": ["食品饮料", "乳业", "调味品", "日化", "零食"],
}

# 五段论阶段编号（顺序反映传导链: 贵金属→基本金属→能源→农业→必选消费）
CYCLE_STAGE_ORDER: dict[str, int] = {
    "precious_metal": 1,
    "base_metal": 2,
    "energy": 3,
    "agriculture": 4,
    "consumer_staples": 5,
}

# 消费/金融防御链
DEFENSIVE_KEYWORDS: list[str] = [
    "银行", "保险", "证券", "白酒", "家电", "房地产", "建材", "钢铁",
]

# 红线禁区关键词（基于classifier.py的REDLINE_ZONES）
REDLINE_KEYWORDS: list[str] = [
    "房地产", "地产", "物业", "家装", "装修", "水泥",
    "百货", "超市", "零售", "餐饮",
    "软件外包", "IT服务", "系统集成",
]

# 卖铲人/上游关键词（基于V10的upstream_keywords）
UPSTREAM_KEYWORDS: dict[str, list[str]] = {
    "chip_upstream": ["半导体设备", "光刻", "晶圆", "封装"],
    "new_energy_upstream": ["锂电池", "充电桩", "储能", "逆变器"],
    "medical_upstream": ["体外诊断", "医疗器械"],
    "telecom_upstream": ["通信", "电信运营商"],
}

# V10产业链敏感性系数 (cycle, liquidity, risk)
CHAIN_SENSITIVITY: dict[str, tuple[float, float, float]] = {
    "tech_chip": (+0.3, +0.7, +0.9),
    "tech_ai_software": (+0.2, +0.8, +1.0),
    "tech_robot_auto": (+0.4, +0.6, +0.8),
    "tech_medical": (-0.1, +0.4, +0.2),
    "tech_new_energy": (+0.3, +0.7, +0.9),
    "tech_telecom": (+0.5, +0.3, +0.6),
    "tech_space_military": (+0.1, +0.2, +0.5),
    "cycle_precious_metal": (-0.5, +0.3, -0.8),
    "cycle_base_metal": (+0.8, +0.5, +0.3),
    "cycle_energy": (+1.0, +0.3, -0.2),
    "cycle_agriculture": (+0.6, +0.1, -0.1),
    "cycle_consumer_staples": (-0.3, -0.1, -0.5),
    "defensive": (+0.5, +0.4, -0.3),
}


# ======================================================================
# 板块分类函数
# ======================================================================


def _classify_board(name: str) -> str:
    """将板块名分类为 tech/cycle/defensive/neutral。"""
    for keywords in TECH_CHAIN_KEYWORDS.values():
        for kw in keywords:
            if kw in name:
                return "tech"
    for keywords in CYCLE_CHAIN_KEYWORDS.values():
        for kw in keywords:
            if kw in name:
                return "cycle"
    for kw in DEFENSIVE_KEYWORDS:
        if kw in name:
            return "defensive"
    return "neutral"


def _get_chain_id(name: str) -> str:
    """将板块名映射到具体产业链（用于同链共振特征）。"""
    for chain_name, keywords in TECH_CHAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in name:
                return f"tech_{chain_name}"
    for chain_name, keywords in CYCLE_CHAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in name:
                return f"cycle_{chain_name}"
    return "other"


def _get_cycle_stage(name: str) -> int:
    """将板块映射到五段论阶段编号（0=非周期板块）。"""
    for stage_name, keywords in CYCLE_CHAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in name:
                return CYCLE_STAGE_ORDER[stage_name]
    return 0


def _is_redline(name: str) -> bool:
    """判断是否红线禁区板块。"""
    return any(kw in name for kw in REDLINE_KEYWORDS)


def _is_upstream(name: str) -> bool:
    """判断是否卖铲人/上游板块。"""
    for keywords in UPSTREAM_KEYWORDS.values():
        for kw in keywords:
            if kw in name:
                return True
    return False


def _get_tech_track(name: str) -> str:
    """将板块映射到具体科技赛道（用于赛道热度特征）。"""
    for track_name, keywords in TECH_CHAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in name:
                return track_name
    return "none"


# ======================================================================
# 特征工程
# ======================================================================


def _build_ts_features(panel: pd.DataFrame) -> pd.DataFrame:
    """构造时间序列特征（与V11相同的基础特征）。"""
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["name", "date"]).reset_index(drop=True)

    num_cols = ["change_pct", "turnover_rate", "rise_count",
                "fall_count", "lead_stock_pct", "momentum_5d"]
    for col in num_cols:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")

    grouped = panel.groupby("name", group_keys=False)

    # 动量因子（多尺度累计收益）
    for w in [3, 5, 10, 20]:
        panel[f"ret_{w}d"] = grouped["change_pct"].transform(
            lambda x: x.rolling(w, min_periods=1).sum()
        )
        panel[f"vol_{w}d"] = grouped["change_pct"].transform(
            lambda x: x.rolling(w, min_periods=2).std()
        )

    # 换手率因子
    if "turnover_rate" in panel.columns:
        panel["to_5d"] = grouped["turnover_rate"].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        panel["to_20d"] = grouped["turnover_rate"].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        panel["to_ratio"] = panel["turnover_rate"] / (panel["to_20d"] + 1e-6)

    # 广度因子
    if "rise_count" in panel.columns and "fall_count" in panel.columns:
        panel["breadth"] = panel["rise_count"] / (
            panel["rise_count"] + panel["fall_count"] + 1e-6
        )
        panel["breadth_5d"] = grouped["breadth"].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )

    # 领涨因子
    if "lead_stock_pct" in panel.columns:
        panel["lead_5d"] = grouped["lead_stock_pct"].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )

    return panel


def _build_linkage_features(panel: pd.DataFrame) -> pd.DataFrame:
    """构造V12原有的12个泽平联动特征。"""
    panel = panel.copy()

    # 板块分类
    panel["_sector_type"] = panel["name"].apply(_classify_board)
    panel["_chain_id"] = panel["name"].apply(_get_chain_id)

    # --- 1. 产业链联动: 每日各类板块的平均涨幅 ---
    for stype in ["tech", "cycle", "defensive"]:
        daily_map = (
            panel[panel["_sector_type"] == stype]
            .groupby("date")["change_pct"]
            .mean()
        )
        panel = panel.merge(
            daily_map.rename(f"{stype}_chain_avg").reset_index(),
            on="date",
            how="left",
        )
        panel[f"{stype}_chain_avg"] = panel[f"{stype}_chain_avg"].fillna(0.0)

    # --- 2. 科技vs周期跷跷板（多尺度spread）---
    panel["tc_spread"] = panel["tech_chain_avg"] - panel["cycle_chain_avg"]
    date_spread = (
        panel.drop_duplicates("date")[["date", "tc_spread"]]
        .sort_values("date")
        .set_index("date")
    )
    for w in [3, 5, 10]:
        rolled = date_spread["tc_spread"].rolling(w, min_periods=1).mean()
        rolled_map = rolled.rename(f"tc_spread_{w}d").reset_index()
        panel = panel.merge(rolled_map, on="date", how="left")

    # --- 3. 同赛道板块共振 ---
    chain_stats = panel.groupby(["date", "_chain_id"])["change_pct"].agg(
        ["mean", "std"]
    ).rename(columns={"mean": "chain_peer_avg", "std": "chain_peer_std"})
    chain_stats = chain_stats.reset_index()
    panel = panel.merge(chain_stats, on=["date", "_chain_id"], how="left")
    panel["chain_peer_avg"] = panel["chain_peer_avg"].fillna(0.0)
    panel["chain_peer_std"] = panel["chain_peer_std"].fillna(0.0)
    panel["chain_relative"] = panel["change_pct"] - panel["chain_peer_avg"]

    # --- 4. 交叉特征 ---
    if "turnover_rate" in panel.columns:
        panel["mom_turnover_cross"] = panel["ret_5d"] * panel["to_5d"]
    if "breadth" in panel.columns:
        panel["vol_breadth_cross"] = panel["vol_5d"] * panel["breadth"]
    panel["mom_divergence"] = panel["ret_3d"] - panel["ret_20d"]

    # 清理临时列（保留_sector_type和_chain_id供后续特征使用）
    panel = panel.drop(columns=["tc_spread"], errors="ignore")

    return panel


def _build_cycle_stage_features(panel: pd.DataFrame) -> pd.DataFrame:
    """构造五段论轮动特征（V13新增）。

    泽平核心理论: 贵金属→基本金属→能源→农业→必选消费，五段依次传导。
    """
    panel = panel.copy()

    # 每日五段各阶段的平均涨幅（温度）
    panel["_cycle_stage"] = panel["name"].apply(_get_cycle_stage)

    for stage_id in range(1, 6):
        stage_daily = (
            panel[panel["_cycle_stage"] == stage_id]
            .groupby("date")["change_pct"]
            .mean()
        )
        panel = panel.merge(
            stage_daily.rename(f"stage_heat_{stage_id}").reset_index(),
            on="date",
            how="left",
        )
        panel[f"stage_heat_{stage_id}"] = panel[f"stage_heat_{stage_id}"].fillna(0.0)

    # 当前最热阶段编号
    stage_cols = [f"stage_heat_{i}" for i in range(1, 6)]
    stage_df = panel.drop_duplicates("date")[["date"] + stage_cols].copy()
    stage_df["hot_stage_id"] = stage_df[stage_cols].values.argmax(axis=1) + 1
    panel = panel.merge(stage_df[["date", "hot_stage_id"]], on="date", how="left")
    panel["hot_stage_id"] = panel["hot_stage_id"].fillna(3).astype(float)

    # 板块距最热阶段的距离（0=最近，4=最远）
    panel["stage_distance"] = panel.apply(
        lambda r: abs(r["_cycle_stage"] - r["hot_stage_id"])
        if r["_cycle_stage"] > 0 else 3.0,  # 非周期板块给中等距离
        axis=1,
    )

    # 领先-滞后信号: stage1(贵金属) vs stage3(能源) spread
    # 贵金属领涨但能源还没涨 → 传导机会
    panel["stage_lead_lag"] = panel["stage_heat_1"] - panel["stage_heat_3"]

    # 清理
    panel = panel.drop(columns=["_cycle_stage"], errors="ignore")

    return panel


def _build_regime_features(panel: pd.DataFrame) -> pd.DataFrame:
    """构造三维度制度信号特征（V13新增）。

    来自V10的三根支柱:
    1. 经济周期: 周期板块动量 - 科技板块动量
    2. 流动性: 全市场换手率水平
    3. 风险偏好: 上涨广度 + 领涨强度
    """
    panel = panel.copy()

    # 确保 _sector_type 存在
    if "_sector_type" not in panel.columns:
        panel["_sector_type"] = panel["name"].apply(_classify_board)

    # --- 经济周期信号: cycle_avg - tech_avg ---
    # 已有 tech_chain_avg 和 cycle_chain_avg
    panel["cycle_signal"] = panel["cycle_chain_avg"] - panel["tech_chain_avg"]

    # --- 流动性信号: 全市场日均换手率 ---
    if "turnover_rate" in panel.columns:
        daily_turnover = panel.groupby("date")["turnover_rate"].mean()
        panel = panel.merge(
            daily_turnover.rename("mkt_avg_turnover").reset_index(),
            on="date",
            how="left",
        )
        # 归一化: 2%为中性，映射到(-1, +1)
        panel["liquidity_signal"] = np.tanh(
            (panel["mkt_avg_turnover"].fillna(2.0) - 2.0) / 1.0
        )
    else:
        panel["mkt_avg_turnover"] = 0.0
        panel["liquidity_signal"] = 0.0

    # --- 风险偏好信号: 上涨广度 + 领涨强度 ---
    if "rise_count" in panel.columns and "fall_count" in panel.columns:
        daily_breadth = panel.groupby("date").apply(
            lambda g: g["rise_count"].sum() / (
                g["rise_count"].sum() + g["fall_count"].sum() + 1e-6
            ),
            include_groups=False,
        )
        panel = panel.merge(
            daily_breadth.rename("mkt_breadth").reset_index(),
            on="date",
            how="left",
        )
        panel["risk_pref_signal"] = np.tanh(
            (panel["mkt_breadth"].fillna(0.5) - 0.5) * 4.0
        )
    else:
        panel["mkt_breadth"] = 0.5
        panel["risk_pref_signal"] = 0.0

    # --- 综合regime分 ---
    panel["regime_score"] = (
        panel["cycle_signal"] + panel["liquidity_signal"] + panel["risk_pref_signal"]
    )

    # --- 板块对三维度的敏感性加权分 ---
    if "_chain_id" not in panel.columns:
        panel["_chain_id"] = panel["name"].apply(_get_chain_id)

    def _sensitivity_score(row: pd.Series) -> float:
        chain_id = row.get("_chain_id", "other")
        sector_type = row.get("_sector_type", "neutral")
        # 查找敏感性系数
        sens = CHAIN_SENSITIVITY.get(chain_id)
        if sens is None and sector_type == "defensive":
            sens = CHAIN_SENSITIVITY.get("defensive")
        if sens is None:
            return 0.0
        signals = np.array([
            row.get("cycle_signal", 0.0),
            row.get("liquidity_signal", 0.0),
            row.get("risk_pref_signal", 0.0),
        ])
        return float(np.dot(sens, signals))

    panel["board_sensitivity_score"] = panel.apply(_sensitivity_score, axis=1)

    # 清理中间列
    panel = panel.drop(columns=["mkt_avg_turnover", "mkt_breadth"],
                       errors="ignore")

    return panel


def _build_track_heat_features(panel: pd.DataFrame) -> pd.DataFrame:
    """构造赛道热度特征（V13新增）。

    基于tech_tracker.py的热度计算逻辑:
    热度 = 动量(40%) + 活跃度(30%) + 广度(30%)
    """
    panel = panel.copy()

    # 每个板块所属的科技赛道
    panel["_tech_track"] = panel["name"].apply(_get_tech_track)

    # 计算每日每赛道热度
    track_heat_records: list[dict[str, object]] = []
    for date_val, day_group in panel.groupby("date"):
        for track_name in TECH_CHAIN_KEYWORDS:
            track_boards = day_group[day_group["_tech_track"] == track_name]
            if track_boards.empty:
                track_heat_records.append({
                    "date": date_val,
                    "_tech_track": track_name,
                    "track_heat": 50.0,
                })
                continue

            # 涨幅动量得分(40%)
            avg_change = track_boards["change_pct"].mean()
            momentum_score = max(0.0, min(100.0, 50.0 + avg_change * 10.0))

            # 活跃度得分(30%)
            if "turnover_rate" in track_boards.columns:
                avg_to = track_boards["turnover_rate"].mean()
                activity_score = min(avg_to * 10, 100.0)
            else:
                activity_score = 50.0

            # 广度得分(30%)
            if ("rise_count" in track_boards.columns
                    and "fall_count" in track_boards.columns):
                total_rise = track_boards["rise_count"].sum()
                total_fall = track_boards["fall_count"].sum()
                total = total_rise + total_fall
                breadth_score = (total_rise / total * 100) if total > 0 else 50.0
            else:
                breadth_score = 50.0

            heat = 0.4 * momentum_score + 0.3 * activity_score + 0.3 * breadth_score
            track_heat_records.append({
                "date": date_val,
                "_tech_track": track_name,
                "track_heat": round(heat, 2),
            })

    track_heat_df = pd.DataFrame(track_heat_records)

    # 合并到panel
    panel = panel.merge(
        track_heat_df,
        on=["date", "_tech_track"],
        how="left",
    )
    panel["track_heat"] = panel["track_heat"].fillna(50.0)

    # 赛道热度排名（每日排名，1=最热）
    panel["track_rank"] = panel.groupby("date")["track_heat"].transform(
        lambda x: x.rank(ascending=False, method="min")
    )

    # 赛道热度的5日移动均值
    panel = panel.sort_values(["name", "date"])
    panel["track_heat_ma5"] = panel.groupby("name")["track_heat"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    # 赛道热度相对全赛道平均
    daily_avg_heat = panel.groupby("date")["track_heat"].mean()
    panel = panel.merge(
        daily_avg_heat.rename("_avg_track_heat").reset_index(),
        on="date",
        how="left",
    )
    panel["track_relative_heat"] = panel["track_heat"] - panel["_avg_track_heat"].fillna(50.0)

    # 清理
    panel = panel.drop(columns=["_tech_track", "_avg_track_heat"], errors="ignore")

    return panel


def _build_upstream_features(panel: pd.DataFrame) -> pd.DataFrame:
    """构造卖铲人/上下游特征（V13新增）。

    泽平招牌理论: "别押赛道，卖铲子" — 下游不确定时押上游增加赢率。
    """
    panel = panel.copy()

    # 是否上游板块
    panel["is_upstream"] = panel["name"].apply(
        lambda n: 1.0 if _is_upstream(n) else 0.0
    )

    # 下游链热度（上游板块受益于下游繁荣）
    # 用同类板块（tech/cycle）的平均涨幅作为下游热度代理
    if "_sector_type" not in panel.columns:
        panel["_sector_type"] = panel["name"].apply(_classify_board)

    # downstream_heat: 上游板块所属大类的日均涨幅
    panel["downstream_heat"] = 0.0
    for stype in ["tech", "cycle"]:
        mask = panel["_sector_type"] == stype
        if mask.any():
            daily_avg = panel[mask].groupby("date")["change_pct"].mean()
            mapping = daily_avg.to_dict()
            panel.loc[mask & (panel["is_upstream"] > 0), "downstream_heat"] = (
                panel.loc[mask & (panel["is_upstream"] > 0), "date"].map(mapping).fillna(0.0)
            )

    # 卖铲人溢价: is_upstream × downstream_heat
    panel["shovel_premium"] = panel["is_upstream"] * panel["downstream_heat"]

    return panel


def _build_redline_features(panel: pd.DataFrame) -> pd.DataFrame:
    """构造红线风险特征（V13新增）。"""
    panel = panel.copy()
    panel["is_redline"] = panel["name"].apply(
        lambda n: 1.0 if _is_redline(n) else 0.0
    )
    return panel


def _build_market_structure_features(panel: pd.DataFrame) -> pd.DataFrame:
    """构造市场分化/集中度特征（V13新增）。"""
    panel = panel.copy()

    if "_sector_type" not in panel.columns:
        panel["_sector_type"] = panel["name"].apply(_classify_board)

    # --- 市场集中度: Herfindahl指数 ---
    # 涨幅集中度: 少数板块涨 = 结构市，多数板块涨 = 普涨
    def _hhi(group: pd.DataFrame) -> float:
        cp = group["change_pct"].values
        cp_positive = np.maximum(cp, 0)
        total = cp_positive.sum()
        if total < 1e-9:
            return 0.0
        shares = cp_positive / total
        return float(np.sum(shares ** 2))

    daily_hhi = panel.groupby("date").apply(_hhi, include_groups=False)
    panel = panel.merge(
        daily_hhi.rename("market_concentration").reset_index(),
        on="date",
        how="left",
    )
    panel["market_concentration"] = panel["market_concentration"].fillna(0.0)

    # --- 科技vs周期广度分歧 ---
    # 不只是涨幅差，还有上涨家数比的差异
    if "rise_count" in panel.columns and "fall_count" in panel.columns:
        for stype in ["tech", "cycle"]:
            sub = panel[panel["_sector_type"] == stype]
            if not sub.empty:
                daily_breadth = sub.groupby("date").apply(
                    lambda g: g["rise_count"].sum() / (
                        g["rise_count"].sum() + g["fall_count"].sum() + 1e-6
                    ),
                    include_groups=False,
                )
                panel = panel.merge(
                    daily_breadth.rename(f"_{stype}_breadth").reset_index(),
                    on="date",
                    how="left",
                )
            else:
                panel[f"_{stype}_breadth"] = 0.5

        panel["_tech_breadth"] = panel["_tech_breadth"].fillna(0.5)
        panel["_cycle_breadth"] = panel["_cycle_breadth"].fillna(0.5)
        panel["tech_cycle_divergence"] = panel["_tech_breadth"] - panel["_cycle_breadth"]
        panel = panel.drop(columns=["_tech_breadth", "_cycle_breadth"], errors="ignore")
    else:
        panel["tech_cycle_divergence"] = 0.0

    # --- 同链板块涨幅离散度 ---
    if "_chain_id" not in panel.columns:
        panel["_chain_id"] = panel["name"].apply(_get_chain_id)

    chain_disp = panel.groupby(["date", "_chain_id"])["change_pct"].std()
    chain_disp = chain_disp.rename("chain_dispersion").reset_index()
    panel = panel.merge(chain_disp, on=["date", "_chain_id"], how="left")
    panel["chain_dispersion"] = panel["chain_dispersion"].fillna(0.0)

    return panel


def _apply_cs_rank(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """对所有数值特征做截面排名归一化。"""
    exclude_cols = {"date", "name", "target_ret", "target_rank", "excess",
                    "_sector_type", "_chain_id"}
    raw_features = [
        c for c in panel.columns
        if c not in exclude_cols
        and not c.startswith("target_")
        and not c.startswith("_")
        and panel[c].dtype in [np.float64, np.float32, np.int64, np.int32,
                               float, int]
    ]

    feat_cols = []
    for col in raw_features:
        rank_col = f"{col}_cs_rank"
        panel[rank_col] = panel.groupby("date")[col].transform(
            lambda x: x.rank(pct=True)
        )
        feat_cols.append(rank_col)

    return panel, feat_cols


def build_training_data(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """构造完整训练数据（V11基础特征 + 全部泽平联动特征 + 目标）。

    Returns:
        (featured_panel, feature_columns)
    """
    # V11基础特征
    panel = _build_ts_features(panel)

    # V12原有联动特征（12个）
    panel = _build_linkage_features(panel)

    # V13新增: 五段论轮动
    panel = _build_cycle_stage_features(panel)

    # V13新增: 三维度制度信号
    panel = _build_regime_features(panel)

    # V13新增: 赛道热度
    panel = _build_track_heat_features(panel)

    # V13新增: 卖铲人/上下游
    panel = _build_upstream_features(panel)

    # V13新增: 红线风险
    panel = _build_redline_features(panel)

    # V13新增: 市场分化
    panel = _build_market_structure_features(panel)

    # 超额收益
    daily_avg = panel.groupby("date")["change_pct"].transform("mean")
    panel["excess"] = panel["change_pct"] - daily_avg

    # 目标: 次日超额收益的截面排名
    grouped = panel.groupby("name", group_keys=False)
    panel["target_ret"] = grouped["excess"].shift(-1)
    panel["target_rank"] = panel.groupby("date")["target_ret"].transform(
        lambda x: x.rank()
    )

    # 截面排名特征
    panel, feat_cols = _apply_cs_rank(panel)

    # 清理临时列
    panel = panel.drop(columns=["_sector_type", "_chain_id"], errors="ignore")

    return panel, feat_cols


def build_inference_features(
    board_df: pd.DataFrame,
    history: list[pd.DataFrame],
) -> tuple[pd.DataFrame, list[str]]:
    """为当日数据构造推理特征。"""
    all_dfs = history + [board_df]
    panel = pd.concat(all_dfs, ignore_index=True)

    # V11基础特征
    panel = _build_ts_features(panel)

    # V12联动特征
    panel = _build_linkage_features(panel)

    # V13新增特征
    panel = _build_cycle_stage_features(panel)
    panel = _build_regime_features(panel)
    panel = _build_track_heat_features(panel)
    panel = _build_upstream_features(panel)
    panel = _build_redline_features(panel)
    panel = _build_market_structure_features(panel)

    # 截面排名
    panel, feat_cols = _apply_cs_rank(panel)

    # 清理临时列
    panel = panel.drop(columns=["_sector_type", "_chain_id"], errors="ignore")

    # 只返回今天
    today_date = pd.to_datetime(board_df["date"].iloc[0])
    result = panel[panel["date"] == today_date].copy()

    return result, feat_cols


# ======================================================================
# 策略类
# ======================================================================


@dataclass
class ZepingPrediction:
    """兼容接口的预测结果。"""
    board_name: str
    score: float = 0.0


@dataclass
class ZepingPredictionResult:
    """兼容接口的预测结果容器。"""
    predictions: list[ZepingPrediction] = field(default_factory=list)


class ZepingLGBMStrategyV13:
    """V13 LightGBM + 全泽平方法论联动特征。

    V13 = V11（截面因子LightGBM）+ 全部泽平宏观特征扩展（~40个特征）。
    固定训练，单1天horizon，无滚动再训练。

    用法:
        v13 = ZepingLGBMStrategyV13()
        v13.fit(panel_train_df)
        result = v13.predict(board_df, top_n=10)
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

        print(f"[V13] 构造截面排名特征 + 泽平联动特征... (panel: {panel_df.shape})")
        featured, feat_cols = build_training_data(panel_df)

        # 去掉没有 target 的行（最后一天）
        featured = featured.dropna(subset=["target_rank"])
        self.feature_cols = feat_cols

        X = featured[self.feature_cols].fillna(0.5)
        y = featured["target_rank"]

        print(f"[V13] 训练 LightGBM... (samples: {len(X)}, features: {len(self.feature_cols)})")
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
        print("[V13] Top15 重要特征:")
        for feat, imp in importance.head(15).items():
            print(f"       {feat}: {imp}")
        print(f"[V13] 总特征数: {len(self.feature_cols)}")

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
        today_featured, infer_feat_cols = build_inference_features(
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


# 兼容别名
ZepingMacroStrategyV13 = ZepingLGBMStrategyV13

__all__ = [
    "ZepingLGBMStrategyV13",
    "ZepingMacroStrategyV13",
    "build_training_data",
    "build_inference_features",
]
