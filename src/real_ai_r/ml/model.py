"""ML 模型模块 — LightGBM 板块热度预测模型

训练和使用 LightGBM 模型预测明日热门板块。
支持模型保存/加载、特征重要性分析。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """模型评估指标。"""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc: float = 0.0
    feature_importance: dict[str, float] = field(default_factory=dict)
    classification_report_text: str = ""


@dataclass
class PredictionResult:
    """单个板块的预测结果。"""

    board_name: str
    hot_probability: float       # 预测为热门的概率 0-1
    predicted_score: float       # 综合预测评分 0-100
    macro_category: str          # 宏观分类
    key_factors: list[str]       # 关键驱动因子
    momentum_1d: float = 0.0
    turnover_rate: float = 0.0


class HotBoardModel:
    """LightGBM 板块热度预测模型。

    结合宏观标签和量化因子，预测明日热门板块。
    """

    DEFAULT_PARAMS = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_estimators": 200,
        "max_depth": 6,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "class_weight": "balanced",
    }

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        params: dict | None = None,
    ) -> None:
        from real_ai_r.ml.features import FeatureEngineer
        self.feature_columns = feature_columns or FeatureEngineer.ALL_FEATURES
        self.params = params or self.DEFAULT_PARAMS.copy()
        self.model = None
        self.is_trained = False
        self._metrics: ModelMetrics | None = None

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "is_hot_next_day",
    ) -> ModelMetrics:
        """训练模型。

        Parameters
        ----------
        df : pd.DataFrame
            含特征列和目标列的训练数据。
        target_col : str
            目标列名。

        Returns
        -------
        ModelMetrics
            训练后的评估指标。
        """
        import lightgbm as lgb

        available_features = [f for f in self.feature_columns if f in df.columns]
        if not available_features:
            logger.error("无可用特征列")
            return ModelMetrics()

        X = df[available_features].fillna(0)
        y = df[target_col].fillna(0).astype(int)

        logger.info(
            "开始训练: %d 样本, %d 特征, 正样本比例 %.1f%%",
            len(X), len(available_features), y.mean() * 100,
        )

        # 时序分割
        tscv = TimeSeriesSplit(n_splits=3)
        oof_preds = np.zeros(len(X))
        oof_proba = np.zeros(len(X))
        val_seen = np.zeros(len(X), dtype=bool)
        models = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(**self.params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
            )

            val_pred = model.predict(X_val)
            val_proba = model.predict_proba(X_val)[:, 1]

            oof_preds[val_idx] = val_pred
            oof_proba[val_idx] = val_proba
            val_seen[val_idx] = True
            models.append(model)

            fold_auc = roc_auc_score(y_val, val_proba) if len(set(y_val)) > 1 else 0
            logger.info("Fold %d AUC: %.4f", fold + 1, fold_auc)

        # 使用最后一个 fold 的模型作为最终模型
        self.model = models[-1]
        self.is_trained = True

        # 计算 OOF 指标（仅统计被验证过的样本）
        val_mask = val_seen
        if val_mask.any():
            y_val_all = y[val_mask]
            pred_val_all = oof_preds[val_mask].astype(int)
            proba_val_all = oof_proba[val_mask]

            has_both = len(set(y_val_all)) > 1
            metrics = ModelMetrics(
                accuracy=accuracy_score(y_val_all, pred_val_all),
                precision=precision_score(y_val_all, pred_val_all, zero_division=0),
                recall=recall_score(y_val_all, pred_val_all, zero_division=0),
                f1=f1_score(y_val_all, pred_val_all, zero_division=0),
                auc=roc_auc_score(y_val_all, proba_val_all) if has_both else 0.0,
                feature_importance=dict(zip(
                    available_features,
                    [float(v) for v in self.model.feature_importances_],
                )),
                classification_report_text=classification_report(
                    y_val_all, pred_val_all, zero_division=0,
                ),
            )
        else:
            metrics = ModelMetrics()

        self._metrics = metrics
        logger.info(
            "训练完成: AUC=%.4f, F1=%.4f, Precision=%.4f, Recall=%.4f",
            metrics.auc, metrics.f1, metrics.precision, metrics.recall,
        )
        return metrics

    def predict(self, df: pd.DataFrame) -> list[PredictionResult]:
        """预测板块热度。

        Parameters
        ----------
        df : pd.DataFrame
            含特征列和 board_name 列的预测数据。

        Returns
        -------
        list[PredictionResult]
            按预测分数排序的预测结果。
        """
        if not self.is_trained or self.model is None:
            logger.warning("模型未训练，使用规则引擎回退")
            return self._fallback_predict(df)

        available_features = [f for f in self.feature_columns if f in df.columns]
        X = df[available_features].fillna(0)

        probas = self.model.predict_proba(X)[:, 1]

        # 获取特征重要性
        importances = dict(zip(
            available_features,
            [float(v) for v in self.model.feature_importances_],
        ))
        top_factors = sorted(importances, key=importances.get, reverse=True)[:5]

        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            board_name = row.get("board_name", f"board_{i}")

            # 确定关键因子
            key_factors = []
            for f in top_factors:
                val = row.get(f, 0)
                if isinstance(val, (int, float)) and abs(val) > 0.01:
                    key_factors.append(f"{f}={val:.2f}")

            # 确定宏观分类
            if row.get("is_tech", 0) == 1:
                macro = "科技主线"
            elif row.get("is_cycle", 0) == 1:
                macro = "周期主线"
            elif row.get("is_redline", 0) == 1:
                macro = "红线禁区"
            else:
                macro = "其他"

            results.append(PredictionResult(
                board_name=board_name,
                hot_probability=float(probas[i]),
                predicted_score=float(probas[i] * 100),
                macro_category=macro,
                key_factors=key_factors[:5],
                momentum_1d=float(row.get("momentum_1d", 0)),
                turnover_rate=float(row.get("turnover_rate", 0)),
            ))

        results.sort(key=lambda r: r.predicted_score, reverse=True)
        return results

    def _fallback_predict(self, df: pd.DataFrame) -> list[PredictionResult]:
        """规则引擎回退预测（模型未训练时使用）。"""
        results = []
        for _, row in df.iterrows():
            board_name = row.get("board_name", "")

            # 简单加权评分
            score = 50.0
            momentum = float(row.get("momentum_1d", 0))
            turnover = float(row.get("turnover_rate", 0))
            is_tech = int(row.get("is_tech", 0))
            is_redline = int(row.get("is_redline", 0))

            score += momentum * 5  # 涨幅贡献
            score += turnover * 2  # 换手率贡献
            score += is_tech * 10  # 科技股加分
            score -= is_redline * 20  # 红线减分
            score = max(0, min(100, score))

            if is_tech:
                macro = "科技主线"
            elif int(row.get("is_cycle", 0)):
                macro = "周期主线"
            elif is_redline:
                macro = "红线禁区"
            else:
                macro = "其他"

            factors = []
            if abs(momentum) > 0.5:
                factors.append(f"动量={momentum:.2f}%")
            if turnover > 3:
                factors.append(f"换手率={turnover:.1f}%")

            results.append(PredictionResult(
                board_name=board_name,
                hot_probability=score / 100,
                predicted_score=score,
                macro_category=macro,
                key_factors=factors,
                momentum_1d=momentum,
                turnover_rate=turnover,
            ))

        results.sort(key=lambda r: r.predicted_score, reverse=True)
        return results

    def save(self, path: str | Path) -> None:
        """保存模型到文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_columns": self.feature_columns,
            "params": self.params,
            "metrics": self._metrics,
        }, path)
        logger.info("模型已保存: %s", path)

    def load(self, path: str | Path) -> None:
        """从文件加载模型。"""
        path = Path(path)
        if not path.exists():
            logger.warning("模型文件不存在: %s", path)
            return

        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data.get("feature_columns", self.feature_columns)
        self.params = data.get("params", self.params)
        self._metrics = data.get("metrics")
        self.is_trained = self.model is not None
        logger.info("模型已加载: %s", path)

    @property
    def metrics(self) -> ModelMetrics | None:
        return self._metrics

    def get_feature_importance(self, top_n: int = 10) -> list[tuple[str, float]]:
        """获取特征重要性排名。"""
        if self._metrics and self._metrics.feature_importance:
            sorted_fi = sorted(
                self._metrics.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            return sorted_fi[:top_n]
        return []
