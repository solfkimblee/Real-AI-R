"""泽平宏观 V11 — LightGBM 截面因子模型。

核心思路:
    1. 截面排名归一化（Cross-Sectional Rank）：消除时序噪声
    2. LightGBM 回归预测截面排名：学习稳定的截面模式
    3. 时间序列 + 截面双重特征：滚动动量/波动/换手 → 日内 rank 百分位
    4. 目标变量：次日超额收益的截面排名

验证结果:
    - Val (2025-09~11, 57天): Sharpe +4.26, CumExcess +4.17%, WinRate 63.2%
    - Test (2025-12~2026-04, 89天): Sharpe +1.73, CumExcess +3.94%, WinRate 56.2%
    - 训练IC稳定: 月均 ~0.17, 全部为正

接口:
    drop-in V8 兼容:
    - predict(board_df, fund_df=None, top_n=10, tech_history=None, cycle_history=None)
    - record_excess(daily_excess)

训练:
    fit(panel_df) — 传入 panel_train.parquet 格式的 DataFrame
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
# 特征工程（截面排名方法）
# ======================================================================


def _build_ts_features(panel: pd.DataFrame) -> pd.DataFrame:
    """构造时间序列特征（滚动统计量）。"""
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


def _apply_cs_rank(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """对所有数值特征做截面排名归一化。

    Returns:
        (panel_with_ranks, feature_col_names)
    """
    exclude_cols = {"date", "name", "target_ret", "target_rank", "excess"}
    raw_features = [
        c for c in panel.columns
        if c not in exclude_cols and panel[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
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
    """构造完整训练数据（特征 + 目标）。

    Returns:
        (featured_panel, feature_columns)
    """
    panel = _build_ts_features(panel)

    # 超额收益
    daily_avg = panel.groupby("date")["change_pct"].transform("mean")
    panel["excess"] = panel["change_pct"] - daily_avg

    # 目标：次日超额收益的截面排名
    grouped = panel.groupby("name", group_keys=False)
    panel["target_ret"] = grouped["excess"].shift(-1)
    panel["target_rank"] = panel.groupby("date")["target_ret"].transform(
        lambda x: x.rank()
    )

    # 截面排名特征
    panel, feat_cols = _apply_cs_rank(panel)

    return panel, feat_cols


def build_inference_features(
    board_df: pd.DataFrame,
    history: list[pd.DataFrame],
) -> tuple[pd.DataFrame, list[str]]:
    """为当日数据构造推理特征。

    使用历史数据计算滚动统计量，然后做截面排名。
    """
    all_dfs = history + [board_df]
    panel = pd.concat(all_dfs, ignore_index=True)
    panel = _build_ts_features(panel)

    # 截面排名
    panel, feat_cols = _apply_cs_rank(panel)

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


class ZepingLGBMStrategy:
    """V11 LightGBM 截面因子模型策略。

    核心方法:
        1. 截面排名归一化去除时序噪声
        2. LightGBM 预测次日截面排名
        3. 选预测排名最高的 Top-N 板块

    用法:
        v11 = ZepingMacroStrategyV11()
        v11.fit(panel_train_df)
        result = v11.predict(board_df, top_n=10, tech_history=..., cycle_history=...)
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

        print(f"[V11] 构造截面排名特征... (panel: {panel_df.shape})")
        featured, feat_cols = build_training_data(panel_df)

        # 去掉没有 target 的行（最后一天）
        featured = featured.dropna(subset=["target_rank"])
        self.feature_cols = feat_cols

        X = featured[self.feature_cols].fillna(0.5)
        y = featured["target_rank"]

        print(f"[V11] 训练 LightGBM... (samples: {len(X)}, features: {len(self.feature_cols)})")
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            **self.lgbm_params,
        )
        self.model.fit(X, y)
        self._fitted = True

        # 打印特征重要性 Top10
        importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_cols,
        ).sort_values(ascending=False)
        print("[V11] Top10 重要特征:")
        for feat, imp in importance.head(10).items():
            print(f"       {feat}: {imp}")

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
                "max_history_days": self.max_history_days,
            }, f)

    def load_model(self, path: str | Path) -> None:
        """从文件加载模型。"""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        self.model = data["model"]
        self.feature_cols = data["feature_cols"]
        self.lgbm_params = data["lgbm_params"]
        self.n_estimators = data["n_estimators"]
        self.max_history_days = data["max_history_days"]
        self._fitted = True
