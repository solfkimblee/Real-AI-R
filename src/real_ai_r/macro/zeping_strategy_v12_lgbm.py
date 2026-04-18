"""泽平宏观 V12 — LightGBM 增强版：滚动再训练 + 多Horizon融合 + 泽平方法论板块联动特征。

三大改进（基于V11 LightGBM截面因子模型）：

1. 滚动再训练（Rolling Retrain）
   - V11: 固定训练一次，模型权重不更新
   - V12: 每 retrain_every 天（默认20）用最近 retrain_window 天数据重新训练
   - 原理: 市场regime变化后因子权重应随之调整
   - 预期: 提升尾部窗口表现，降低MaxDD

2. 多Horizon融合（Multi-Horizon Ensemble）
   - V11: 只预测1天forward return
   - V12: 同时训练1d/3d/5d三个horizon的LightGBM，加权ensemble
   - 原理: 短期（reversal）和中期（momentum）信号互补
   - 预期: 提升稳定性，降低夏普CV

3. 特征扩展 — 泽平宏观方法论板块联动
   - V11: 20个纯价量截面特征
   - V12 新增:
     a) 产业链联动: 科技链/周期链/消费链的平均动量作为全局特征
     b) 科技vs周期跷跷板: tech_avg - cycle_avg 的3d/5d/10d spread
     c) 同赛道板块共振: 同分类板块的涨跌一致性（联动强度）
     d) 交叉特征: momentum×turnover（放量动量）, volatility×breadth（波动广度）
   - 预期: IC从0.17提升到0.22+

接口:
    drop-in V8 兼容:
    - fit(panel_df) — 初始训练
    - predict(board_df, fund_df=None, top_n=10, tech_history=None, cycle_history=None)
    - record_excess(daily_excess)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore[assignment]


# ======================================================================
# 泽平宏观板块分类（用于联动特征）
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

# 周期产业链关键词（基于classifier.py的CYCLE_STAGES）
CYCLE_CHAIN_KEYWORDS: dict[str, list[str]] = {
    "precious_metal": ["黄金", "白银", "贵金属"],
    "base_metal": ["有色金属", "稀有金属", "稀土", "铜", "铝", "锌",
                   "小金属"],
    "energy": ["煤炭", "石油", "天然气", "油气"],
    "agriculture": ["农业", "化肥", "农药", "种子", "饲料", "生猪", "养殖"],
    "consumer_staples": ["食品饮料", "乳业", "调味品", "日化", "零食"],
}

# 消费/金融防御链
DEFENSIVE_KEYWORDS: list[str] = [
    "银行", "保险", "证券", "白酒", "家电", "房地产", "建材", "钢铁",
]


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


# ======================================================================
# 特征工程（V11基础 + V12扩展）
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
    """构造泽平宏观板块联动特征（V12新增）。

    新增特征:
    1. 产业链联动 — tech_chain_avg/cycle_chain_avg/defensive_chain_avg
    2. 科技vs周期跷跷板 — tech_cycle_spread_Nd
    3. 同赛道共振 — chain_peer_avg, chain_peer_std
    4. 交叉特征 — momentum×turnover, volatility×breadth
    """
    panel = panel.copy()

    # 板块分类
    panel["_sector_type"] = panel["name"].apply(_classify_board)
    panel["_chain_id"] = panel["name"].apply(_get_chain_id)

    # --- 1. 产业链联动: 每日各类板块的平均涨幅 ---
    for stype in ["tech", "cycle", "defensive"]:
        daily_avg = panel[panel["_sector_type"] == stype].groupby("date")[
            "change_pct"
        ].transform("mean")
        # 广播到全部行
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
    # 需要按日期排序后做滚动
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
    # 同链板块的日均涨幅和标准差
    chain_stats = panel.groupby(["date", "_chain_id"])["change_pct"].agg(
        ["mean", "std"]
    ).rename(columns={"mean": "chain_peer_avg", "std": "chain_peer_std"})
    chain_stats = chain_stats.reset_index()
    panel = panel.merge(chain_stats, on=["date", "_chain_id"], how="left")
    panel["chain_peer_avg"] = panel["chain_peer_avg"].fillna(0.0)
    panel["chain_peer_std"] = panel["chain_peer_std"].fillna(0.0)

    # 板块相对同链的强弱
    panel["chain_relative"] = panel["change_pct"] - panel["chain_peer_avg"]

    # --- 4. 交叉特征 ---
    # 放量动量: momentum × turnover（放量上涨 vs 缩量上涨）
    if "turnover_rate" in panel.columns:
        panel["mom_turnover_cross"] = panel["ret_5d"] * panel["to_5d"]

    # 波动广度: volatility × breadth（高波动+高广度=确认性强趋势）
    if "breadth" in panel.columns:
        panel["vol_breadth_cross"] = panel["vol_5d"] * panel["breadth"]

    # 动量分歧: 短期vs长期动量（反转信号）
    panel["mom_divergence"] = panel["ret_3d"] - panel["ret_20d"]

    # 清理临时列
    panel = panel.drop(columns=["_sector_type", "_chain_id", "tc_spread"],
                       errors="ignore")

    return panel


def _apply_cs_rank(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """对所有数值特征做截面排名归一化。"""
    exclude_cols = {
        "date", "name", "target_ret_1d", "target_ret_3d", "target_ret_5d",
        "target_rank_1d", "target_rank_3d", "target_rank_5d",
        "target_ret", "target_rank", "excess",
    }
    raw_features = [
        c for c in panel.columns
        if c not in exclude_cols
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


def build_training_data(
    panel: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5),
) -> tuple[pd.DataFrame, list[str]]:
    """构造完整训练数据（V11基础特征 + V12联动特征 + 多horizon目标）。

    Returns:
        (featured_panel, feature_columns)
    """
    panel = _build_ts_features(panel)
    panel = _build_linkage_features(panel)

    # 超额收益
    daily_avg = panel.groupby("date")["change_pct"].transform("mean")
    panel["excess"] = panel["change_pct"] - daily_avg

    # 多horizon目标: N日累计超额收益的截面排名
    grouped = panel.groupby("name", group_keys=False)
    for h in horizons:
        if h == 1:
            panel[f"target_ret_{h}d"] = grouped["excess"].shift(-1)
        else:
            # N日累计超额
            panel[f"target_ret_{h}d"] = grouped["excess"].transform(
                lambda x: x.rolling(h, min_periods=1).sum().shift(-h)
            )
        panel[f"target_rank_{h}d"] = panel.groupby("date")[
            f"target_ret_{h}d"
        ].transform(lambda x: x.rank())

    # 截面排名特征
    panel, feat_cols = _apply_cs_rank(panel)

    return panel, feat_cols


def build_inference_features(
    board_df: pd.DataFrame,
    history: list[pd.DataFrame],
) -> tuple[pd.DataFrame, list[str]]:
    """为当日数据构造推理特征（含V12联动特征）。"""
    all_dfs = history + [board_df]
    panel = pd.concat(all_dfs, ignore_index=True)
    panel = _build_ts_features(panel)
    panel = _build_linkage_features(panel)
    panel, feat_cols = _apply_cs_rank(panel)

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


class ZepingLGBMStrategyV12:
    """V12 LightGBM 增强策略 — 滚动再训练 + 多Horizon + 泽平联动特征。

    用法:
        v12 = ZepingLGBMStrategyV12()
        v12.fit(panel_train_df)
        result = v12.predict(board_df, top_n=10)
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_history_days: int = 30,
        retrain_every: int = 20,
        retrain_window: int = 250,
        horizons: tuple[int, ...] = (1, 3, 5),
        horizon_weights: tuple[float, ...] = (0.5, 0.3, 0.2),
        lgbm_params: dict[str, Any] | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_history_days = max_history_days
        self.retrain_every = retrain_every
        self.retrain_window = retrain_window
        self.horizons = horizons
        self.horizon_weights = horizon_weights
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
        # 多horizon模型: {horizon: LGBMRegressor}
        self.models: dict[int, Any] = {}
        self.feature_cols: list[str] = []
        self._history: list[pd.DataFrame] = []
        self._accumulated_panel: pd.DataFrame | None = None
        self._excess_history: list[float] = []
        self._days_since_retrain: int = 0
        self._fitted = False

    def _train_models(self, panel_df: pd.DataFrame, verbose: bool = True) -> None:
        """在给定面板上训练多horizon LightGBM模型。"""
        if lgb is None:
            raise ImportError("lightgbm not installed")

        featured, feat_cols = build_training_data(panel_df, self.horizons)
        self.feature_cols = feat_cols

        for h in self.horizons:
            target_col = f"target_rank_{h}d"
            train_df = featured.dropna(subset=[target_col])

            X = train_df[self.feature_cols].fillna(0.5)
            y = train_df[target_col]

            if len(X) < 100:
                if verbose:
                    print(f"[V12] {h}d horizon: 样本不足({len(X)}), 跳过")
                continue

            model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                **self.lgbm_params,
            )
            model.fit(X, y)
            self.models[h] = model

            if verbose:
                print(f"[V12] {h}d horizon: 训练完成 (samples={len(X)}, "
                      f"features={len(self.feature_cols)})")

        self._fitted = bool(self.models)

        # 打印特征重要性 Top10（用1d模型）
        if verbose and 1 in self.models:
            importance = pd.Series(
                self.models[1].feature_importances_,
                index=self.feature_cols,
            ).sort_values(ascending=False)
            print("[V12] Top10 重要特征 (1d模型):")
            for feat, imp in importance.head(10).items():
                print(f"       {feat}: {imp}")

    def fit(self, panel_df: pd.DataFrame) -> None:
        """在训练面板上训练多horizon LightGBM模型。"""
        print(f"[V12] 构造截面排名+联动特征... (panel: {panel_df.shape})")
        self._train_models(panel_df, verbose=True)

        # 保存最后 max_history_days 天数据作为初始历史
        panel_df = panel_df.copy()
        panel_df["date"] = pd.to_datetime(panel_df["date"])
        dates = sorted(panel_df["date"].unique())
        for d in dates[-self.max_history_days:]:
            self._history.append(
                panel_df[panel_df["date"] == d].reset_index(drop=True)
            )

        # 保存完整训练数据用于滚动再训练
        self._accumulated_panel = panel_df.copy()
        self._days_since_retrain = 0

    def _maybe_retrain(self) -> None:
        """检查是否需要滚动再训练。"""
        if (
            self._accumulated_panel is None
            or self._days_since_retrain < self.retrain_every
        ):
            return

        # 取最近 retrain_window 天数据重新训练
        panel = self._accumulated_panel.copy()
        panel["date"] = pd.to_datetime(panel["date"])
        dates = sorted(panel["date"].unique())
        if len(dates) > self.retrain_window:
            cutoff = dates[-self.retrain_window]
            panel = panel[panel["date"] >= cutoff]

        print(f"[V12] 滚动再训练: {len(dates)}天数据 → "
              f"截取{panel['date'].nunique()}天")
        self._train_models(panel, verbose=False)
        self._days_since_retrain = 0

    def predict(
        self,
        board_df: pd.DataFrame | None = None,
        fund_df: pd.DataFrame | None = None,
        top_n: int = 10,
        tech_history: list[float] | None = None,
        cycle_history: list[float] | None = None,
    ) -> ZepingPredictionResult:
        """预测当日 Top-N 板块（多horizon加权融合）。"""
        if board_df is None or board_df.empty:
            return ZepingPredictionResult(predictions=[])

        if not self._fitted or not self.models:
            return ZepingPredictionResult(predictions=[])

        # 滚动再训练检查
        self._maybe_retrain()

        # 构造今日特征（含联动特征）
        today_featured, infer_feat_cols = build_inference_features(
            board_df=board_df,
            history=self._history[-self.max_history_days:],
        )

        if today_featured.empty:
            self._history.append(board_df.copy())
            self._days_since_retrain += 1
            return ZepingPredictionResult(predictions=[])

        # 对齐特征列
        for col in self.feature_cols:
            if col not in today_featured.columns:
                today_featured[col] = 0.5

        X = today_featured[self.feature_cols].fillna(0.5)

        # 多horizon加权融合
        combined_scores = np.zeros(len(X))
        total_weight = 0.0
        for h, w in zip(self.horizons, self.horizon_weights):
            if h in self.models:
                scores = self.models[h].predict(X)
                combined_scores += w * scores
                total_weight += w

        if total_weight > 0:
            combined_scores /= total_weight

        today_featured = today_featured.copy()
        today_featured["pred_score"] = combined_scores

        # 排序选 Top-N
        top_boards = today_featured.nlargest(top_n, "pred_score")[
            ["name", "pred_score"]
        ]

        predictions = [
            ZepingPrediction(board_name=row["name"], score=row["pred_score"])
            for _, row in top_boards.iterrows()
        ]

        # 记录今日数据到历史 + 累积训练数据
        self._history.append(board_df.copy())
        if len(self._history) > self.max_history_days:
            self._history = self._history[-self.max_history_days:]

        if self._accumulated_panel is not None:
            self._accumulated_panel = pd.concat(
                [self._accumulated_panel, board_df], ignore_index=True
            )

        self._days_since_retrain += 1

        return ZepingPredictionResult(predictions=predictions)

    def record_excess(self, daily_excess: float) -> None:
        """记录每日超额收益。"""
        self._excess_history.append(daily_excess)
