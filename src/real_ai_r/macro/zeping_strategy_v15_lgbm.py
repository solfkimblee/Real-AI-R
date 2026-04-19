"""泽平宏观 V15 — V11 + consensus_divergence（共识度分歧）精简版。

V15 = V11（20个纯价量截面特征）+ 1个共识度分歧特征 = 21个特征

设计哲学:
    V14消融发现 consensus_divergence 是所有新增特征中最有价值的信号
    （LightGBM重要性排第5, 106），来源于泽平核心交易哲学:
    "市场一旦形成共识，反着来"

    consensus_divergence = sign(ret_3d) - sign(ret_10d)
    - 短多中空 (+2): 短期反弹但中期趋势仍下 → 可能见顶
    - 短空中多 (-2): 短期回调但中期趋势仍上 → 可能见底（买入机会）
    - 同向 (0): 短中期一致，无分歧

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

# 复用V11的全部基础设施
from real_ai_r.macro.zeping_strategy_v11_lgbm import (
    _build_ts_features,
    _apply_cs_rank,
    build_inference_features as v11_build_inference_features,
    ZepingPrediction,
    ZepingPredictionResult,
)


# ======================================================================
# V15 新增特征: consensus_divergence
# ======================================================================


def _build_consensus_divergence(panel: pd.DataFrame) -> pd.DataFrame:
    """构造共识度分歧特征（V15核心新增）。

    泽平: "市场一旦形成共识，反着来"
    短期动量方向 vs 中期动量方向不一致 = 分歧信号

    - ret_3d 和 ret_10d 由 _build_ts_features 产生
    - consensus_divergence = sign(ret_3d) - sign(ret_10d)
    """
    panel = panel.copy()

    if "ret_3d" in panel.columns and "ret_10d" in panel.columns:
        panel["consensus_divergence"] = (
            np.sign(panel["ret_3d"]) - np.sign(panel["ret_10d"])
        )
    else:
        panel["consensus_divergence"] = 0.0

    return panel


# ======================================================================
# 训练数据构造
# ======================================================================


def build_training_data_v15(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """构造V15完整训练数据（V11特征 + consensus_divergence + 目标）。

    Returns:
        (featured_panel, feature_columns)
    """
    # V11基础特征（20个时序/价量特征）
    panel = _build_ts_features(panel)

    # V15新增: 共识度分歧（1个特征）
    panel = _build_consensus_divergence(panel)

    # 目标变量
    daily_avg = panel.groupby("date")["change_pct"].transform("mean")
    panel["excess"] = panel["change_pct"] - daily_avg

    grouped = panel.groupby("name", group_keys=False)
    panel["target_ret"] = grouped["excess"].shift(-1)
    panel["target_rank"] = panel.groupby("date")["target_ret"].transform(
        lambda x: x.rank()
    )

    # 截面排名归一化
    panel, feat_cols = _apply_cs_rank(panel)

    return panel, feat_cols


def build_inference_features_v15(
    board_df: pd.DataFrame,
    history: list[pd.DataFrame],
) -> tuple[pd.DataFrame, list[str]]:
    """为当日数据构造V15推理特征。

    在V11推理管线基础上追加consensus_divergence。
    """
    if not history:
        return pd.DataFrame(), []

    # 拼接历史+今日，跑完整特征管线
    all_dfs = list(history) + [board_df]
    panel = pd.concat(all_dfs, ignore_index=True)
    panel = _build_ts_features(panel)
    panel = _build_consensus_divergence(panel)

    # 截面排名
    panel, feat_cols = _apply_cs_rank(panel)

    # 只返回今天
    today_date = pd.to_datetime(board_df["date"].iloc[0])
    result = panel[panel["date"] == today_date].copy()

    return result, feat_cols


# ======================================================================
# 策略类
# ======================================================================


class ZepingLGBMStrategyV15:
    """V15 LightGBM: V11 + consensus_divergence 精简版。

    21个特征 = V11的20个纯价量特征 + 1个共识度分歧特征。
    固定训练，单horizon，不做滚动再训练。
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

        print(f"[V15] 构造截面排名特征 + consensus_divergence... (panel: {panel_df.shape})")
        featured, feat_cols = build_training_data_v15(panel_df)

        featured = featured.dropna(subset=["target_rank"])
        self.feature_cols = feat_cols

        X = featured[self.feature_cols].fillna(0.5)
        y = featured["target_rank"]

        print(f"[V15] 训练 LightGBM... (samples: {len(X)}, features: {len(self.feature_cols)})")
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
        print("[V15] Top15 重要特征:")
        for feat, imp in importance.head(15).items():
            print(f"       {feat}: {imp}")
        print(f"[V15] 总特征: {len(self.feature_cols)}")

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

        # 构造今日特征（V15: V11 + consensus_divergence）
        today_featured, infer_feat_cols = build_inference_features_v15(
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
        top_boards = today_featured.nlargest(top_n, "pred_score")[
            ["name", "pred_score"]
        ]

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
