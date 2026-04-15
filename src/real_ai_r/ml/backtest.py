"""模型回测与评估模块

对板块热度预测模型进行历史回测，评估预测效果：
- 滚动窗口回测：每天用过去 N 天训练，预测次日
- 计算命中率、收益率、信息系数等指标
- 生成回测报告
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from real_ai_r.ml.features import FeatureEngineer
from real_ai_r.ml.model import HotBoardModel, ModelMetrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestDay:
    """单日回测结果。"""

    date: str
    predicted_hot: list[str]     # 预测的热门板块
    actual_hot: list[str]        # 实际的热门板块
    hit_count: int               # 命中数
    precision: float             # 当日精确率
    predicted_avg_return: float  # 预测板块平均收益
    market_avg_return: float     # 市场平均收益
    excess_return: float         # 超额收益


@dataclass
class BacktestReport:
    """完整回测报告。"""

    total_days: int = 0
    avg_precision: float = 0.0        # 平均精确率
    avg_hit_rate: float = 0.0         # 平均命中率
    avg_excess_return: float = 0.0    # 平均超额收益
    cumulative_return: float = 0.0    # 累计收益率
    max_drawdown: float = 0.0         # 最大回撤
    sharpe_ratio: float = 0.0         # 夏普比率
    win_rate: float = 0.0             # 胜率（超额>0的天数占比）
    model_metrics: ModelMetrics | None = None
    daily_results: list[BacktestDay] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


class ModelBacktester:
    """模型回测器。

    使用滚动窗口方式，对预测模型进行历史回测评估。
    """

    def __init__(
        self,
        train_window: int = 30,
        top_n: int = 10,
        retrain_every: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        train_window : int
            训练窗口大小（天数）。
        top_n : int
            每日预测的热门板块数量。
        retrain_every : int
            每隔多少天重新训练模型。
        """
        self.train_window = train_window
        self.top_n = top_n
        self.retrain_every = retrain_every

    def run(self, feature_df: pd.DataFrame) -> BacktestReport:
        """执行滚动窗口回测。

        Parameters
        ----------
        feature_df : pd.DataFrame
            含特征列、target 列、date 列、board_name 列的完整数据。

        Returns
        -------
        BacktestReport
            回测结果报告。
        """
        if feature_df.empty or "date" not in feature_df.columns:
            return BacktestReport()

        dates = sorted(feature_df["date"].unique())
        if len(dates) < self.train_window + 5:
            logger.warning("数据天数不足: %d 天, 需要至少 %d 天", len(dates), self.train_window + 5)
            return BacktestReport()

        engineer = FeatureEngineer()
        feature_cols = [f for f in engineer.ALL_FEATURES if f in feature_df.columns]

        daily_results: list[BacktestDay] = []
        equity = [1.0]
        model = HotBoardModel(feature_columns=feature_cols)
        last_train_day = -self.retrain_every  # 强制首次训练

        for i in range(self.train_window, len(dates) - 1):
            current_date = dates[i]
            next_date = dates[i + 1]

            # 训练窗口
            train_dates = dates[max(0, i - self.train_window):i]
            train_data = feature_df[feature_df["date"].isin(train_dates)]

            # 是否需要重新训练
            if i - last_train_day >= self.retrain_every:
                if len(train_data) > 50:
                    model.train(train_data)
                    last_train_day = i

            # 当日截面数据
            today_data = feature_df[feature_df["date"] == current_date]
            if today_data.empty:
                continue

            # 预测
            predictions = model.predict(today_data)
            predicted_hot = [p.board_name for p in predictions[:self.top_n]]

            # 实际次日热门
            next_data = feature_df[feature_df["date"] == next_date]
            if next_data.empty:
                continue

            actual_hot_df = next_data.nlargest(self.top_n, "momentum_1d")
            actual_hot = actual_hot_df["board_name"].tolist()

            # 计算命中
            hit_set = set(predicted_hot) & set(actual_hot)
            hit_count = len(hit_set)
            precision = hit_count / len(predicted_hot) if predicted_hot else 0.0

            # 计算收益
            predicted_returns = next_data[
                next_data["board_name"].isin(predicted_hot)
            ]["momentum_1d"]
            predicted_avg = predicted_returns.mean() if not predicted_returns.empty else 0.0
            market_avg = next_data["momentum_1d"].mean()
            excess = predicted_avg - market_avg

            daily_results.append(BacktestDay(
                date=str(current_date)[:10],
                predicted_hot=predicted_hot,
                actual_hot=actual_hot,
                hit_count=hit_count,
                precision=precision,
                predicted_avg_return=round(predicted_avg, 4),
                market_avg_return=round(market_avg, 4),
                excess_return=round(excess, 4),
            ))

            # 更新净值
            daily_return = predicted_avg / 100  # 转为小数
            equity.append(equity[-1] * (1 + daily_return))

        if not daily_results:
            return BacktestReport()

        # 汇总统计
        precisions = [d.precision for d in daily_results]
        excess_returns = [d.excess_return for d in daily_results]
        win_days = sum(1 for e in excess_returns if e > 0)

        # 最大回撤
        peak = equity[0]
        max_dd = 0.0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd

        # 夏普比率
        daily_rets = np.diff(equity) / np.array(equity[:-1])
        sharpe = 0.0
        if len(daily_rets) > 1 and np.std(daily_rets) > 0:
            sharpe = float(np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252))

        report = BacktestReport(
            total_days=len(daily_results),
            avg_precision=float(np.mean(precisions)),
            avg_hit_rate=float(np.mean(precisions)),
            avg_excess_return=float(np.mean(excess_returns)),
            cumulative_return=float((equity[-1] / equity[0] - 1) * 100),
            max_drawdown=float(max_dd * 100),
            sharpe_ratio=round(sharpe, 2),
            win_rate=win_days / len(daily_results) if daily_results else 0.0,
            model_metrics=model.metrics,
            daily_results=daily_results,
            equity_curve=equity,
        )

        logger.info(
            "回测完成: %d天, 命中率=%.1f%%, 超额收益=%.2f%%, 夏普=%.2f",
            report.total_days, report.avg_precision * 100,
            report.avg_excess_return, report.sharpe_ratio,
        )

        return report
