"""V9 走步回测系统 — 板块轮动日频回测框架。

与 engine/backtest.py 的单股回测不同，V9 回测面对的是一个**组合问题**：
每日选一组板块 + 权重，追踪实现收益。

输入: 面板数据 DataFrame
    必须列: date, name, change_pct
    可选列: turnover_rate, rise_count, fall_count, lead_stock_pct,
           main_net_inflow 等（传递给因子）

参数:
    lookforward_days: 持仓期（日）；1 = 日频再平衡
    benchmark_col: 基准板块名，或 'equal' 用截面等权

输出: V9BacktestResult，包含:
    - equity_curve: 策略净值序列
    - benchmark_curve: 基准净值
    - daily_returns, daily_excess
    - positions_history: 每日持仓快照
    - metrics: 年化、夏普、最大回撤、换手等
    - factor_ic_history: 因子 IC 诊断
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from real_ai_r.v9.engine import V9Config, V9Engine, V9Prediction


# ----------------------------------------------------------------------
# 结果对象
# ----------------------------------------------------------------------


@dataclass
class V9BacktestResult:
    """V9 回测结果。"""

    config: V9Config
    dates: list[str]
    equity_curve: pd.Series                     # 策略净值
    benchmark_curve: pd.Series                  # 基准净值
    daily_returns: pd.Series
    daily_excess: pd.Series
    positions_history: list[dict[str, float]]   # 每日板块权重
    turnover_series: pd.Series
    n_positions_series: pd.Series
    factor_ic_history: dict[str, list[float]] = field(default_factory=dict)
    factor_weight_history: list[dict[str, float]] = field(default_factory=list)
    regime_posterior_history: list[list[float] | None] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"V9 Backtest Summary ({len(self.dates)} days)",
            "-" * 50,
            f"  Total Return:    {m.get('total_return', 0):.2%}",
            f"  Annual Return:   {m.get('annual_return', 0):.2%}",
            f"  Annual Excess:   {m.get('annual_excess', 0):.2%}",
            f"  Volatility:      {m.get('volatility', 0):.2%}",
            f"  Sharpe:          {m.get('sharpe', 0):.2f}",
            f"  Info Ratio:      {m.get('info_ratio', 0):.2f}",
            f"  Max Drawdown:    {m.get('max_drawdown', 0):.2%}",
            f"  Calmar:          {m.get('calmar', 0):.2f}",
            f"  Avg Turnover:    {m.get('avg_turnover', 0):.2%}",
            f"  Win Rate (daily):{m.get('win_rate', 0):.2%}",
            f"  #Positions avg:  {m.get('avg_positions', 0):.1f}",
        ]
        return "\n".join(lines)


# ----------------------------------------------------------------------
# 回测器
# ----------------------------------------------------------------------


@dataclass
class V9BacktestRunner:
    """V9 走步回测执行器。

    用法:
        runner = V9BacktestRunner(config=V9Config(...))
        result = runner.run(panel_df, start="2024-01-01", end="2024-12-31")
        print(result.summary())
    """

    config: V9Config = field(default_factory=V9Config)
    transaction_cost_bps: float = 5.0          # 单边交易成本（bps）
    warmup_days: int = 20                      # 冷启动天数（此期间只累积状态，不算收益）
    benchmark: str = "equal"                    # 'equal' or 板块名
    regime_feature_fn: Any = None              # 自定义: 从截面 df 计算市场特征的函数
    extra_signals_fn: Any = None               # 自定义: 从截面 df 提取 extra_signals

    def run(
        self,
        panel_df: pd.DataFrame,
        start: str | None = None,
        end: str | None = None,
        engine: V9Engine | None = None,
    ) -> V9BacktestResult:
        """执行回测。

        panel_df 必须含: date, name, change_pct
        """
        df = panel_df.copy()
        for col in ("date", "name", "change_pct"):
            if col not in df.columns:
                raise ValueError(f"panel_df missing column: {col}")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "name"]).reset_index(drop=True)

        if start is not None:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df["date"] <= pd.to_datetime(end)]

        unique_dates = sorted(df["date"].unique())
        if len(unique_dates) < 2:
            raise ValueError("Need at least 2 dates for backtest")

        if engine is None:
            engine = V9Engine(config=self.config)

        # 输出容器
        daily_ret_list: list[float] = []
        daily_excess_list: list[float] = []
        bench_ret_list: list[float] = []
        positions_list: list[dict[str, float]] = []
        turnover_list: list[float] = []
        n_pos_list: list[int] = []
        factor_weights_history: list[dict[str, float]] = []
        regime_post_history: list[list[float] | None] = []
        kept_dates: list[str] = []

        prev_weights: dict[str, float] = {}

        for i, date in enumerate(unique_dates[:-1]):
            slice_df = df[df["date"] == date].reset_index(drop=True)
            next_df = df[df["date"] == unique_dates[i + 1]].reset_index(drop=True)

            if slice_df.empty or next_df.empty:
                continue

            # --- 构造 regime 特征 & extra_signals（可选） ---
            regime_feat: np.ndarray | None = None
            if self.regime_feature_fn is not None:
                try:
                    regime_feat = self.regime_feature_fn(slice_df)
                except Exception:
                    regime_feat = None
            else:
                regime_feat = self._default_regime_features(slice_df)

            extra: dict[str, dict[str, float]] = {}
            if self.extra_signals_fn is not None:
                try:
                    extra = self.extra_signals_fn(slice_df)
                except Exception:
                    extra = {}

            # --- 预测 ---
            pred: V9Prediction = engine.predict(
                board_df=slice_df,
                extra_signals=extra,
                regime_features=regime_feat,
                as_of=str(date.date()),
            )

            weights = pred.weights
            positions_list.append(
                {k: float(v) for k, v in weights.items() if v > 1e-6},
            )
            n_pos_list.append(int((weights > 1e-6).sum()))
            turnover_list.append(float(pred.turnover))
            factor_weights_history.append(dict(pred.factor_weights))
            regime_post_history.append(
                pred.regime_posterior.tolist()
                if pred.regime_posterior is not None else None,
            )

            # --- 计算下一期组合实现收益 ---
            realized_map = dict(
                zip(
                    next_df["name"].astype(str).values,
                    pd.to_numeric(next_df["change_pct"], errors="coerce")
                    .fillna(0.0).values,
                    strict=False,
                ),
            )
            portfolio_ret = 0.0
            for bn, w in weights.items():
                if w > 1e-6:
                    portfolio_ret += float(w) * float(realized_map.get(bn, 0.0))

            # --- 换手成本 ---
            cost = float(pred.turnover) * self.transaction_cost_bps / 10000.0 * 100.0
            # 注意 change_pct 单位是 %，这里 cost 也以 % 单位表示
            portfolio_ret -= cost

            # --- 基准 ---
            if self.benchmark == "equal":
                bench_ret = float(
                    pd.to_numeric(next_df["change_pct"], errors="coerce")
                    .fillna(0.0).mean(),
                )
            else:
                br = realized_map.get(self.benchmark)
                bench_ret = float(br) if br is not None else 0.0

            # --- 反馈 ---
            engine.update_feedback(
                realized_returns=realized_map,
                benchmark_return=bench_ret,
            )

            # --- 收益记录 (warmup 期跳过) ---
            if i >= self.warmup_days:
                daily_ret_list.append(portfolio_ret)
                bench_ret_list.append(bench_ret)
                daily_excess_list.append(portfolio_ret - bench_ret)
                kept_dates.append(str(date.date()))

            prev_weights = dict(weights)

        # --- 构造序列 & 指标 ---
        daily_ret = pd.Series(daily_ret_list, index=kept_dates, dtype=float) / 100.0
        bench_ret = pd.Series(bench_ret_list, index=kept_dates, dtype=float) / 100.0
        excess = pd.Series(daily_excess_list, index=kept_dates, dtype=float) / 100.0

        equity = (1.0 + daily_ret).cumprod() if len(daily_ret) > 0 else pd.Series(dtype=float)
        bench_equity = (1.0 + bench_ret).cumprod() if len(bench_ret) > 0 else pd.Series(dtype=float)

        metrics = self._compute_metrics(
            daily_ret, excess, turnover_list[self.warmup_days:], n_pos_list[self.warmup_days:],
        )

        return V9BacktestResult(
            config=self.config,
            dates=kept_dates,
            equity_curve=equity,
            benchmark_curve=bench_equity,
            daily_returns=daily_ret,
            daily_excess=excess,
            positions_history=positions_list[self.warmup_days:],
            turnover_series=pd.Series(
                turnover_list[self.warmup_days:], index=kept_dates,
            ) if kept_dates else pd.Series(dtype=float),
            n_positions_series=pd.Series(
                n_pos_list[self.warmup_days:], index=kept_dates,
            ) if kept_dates else pd.Series(dtype=float),
            factor_ic_history={
                k: list(v) for k, v in engine.state.factor_ic_history.items()
            },
            factor_weight_history=factor_weights_history[self.warmup_days:],
            regime_posterior_history=regime_post_history[self.warmup_days:],
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _default_regime_features(slice_df: pd.DataFrame) -> np.ndarray:
        """默认市场特征: [均涨, 波动, 分化, 上涨比] 四维。"""
        cp = pd.to_numeric(slice_df["change_pct"], errors="coerce").fillna(0.0)
        mean_ret = float(cp.mean())
        vol = float(cp.std(ddof=0))
        # 分化: 涨幅 top20% 与 bottom20% 的差
        if len(cp) >= 10:
            sorted_cp = np.sort(cp.values)
            q_top = sorted_cp[-max(1, len(cp) // 5):]
            q_bot = sorted_cp[:max(1, len(cp) // 5)]
            disp = float(q_top.mean() - q_bot.mean())
        else:
            disp = 0.0
        rise = 0.5
        if "rise_count" in slice_df.columns and "fall_count" in slice_df.columns:
            r = pd.to_numeric(slice_df["rise_count"], errors="coerce").fillna(0.0)
            f = pd.to_numeric(slice_df["fall_count"], errors="coerce").fillna(0.0)
            denom = (r + f).sum()
            if denom > 0:
                rise = float(r.sum() / denom)
        return np.array([mean_ret, vol, disp, rise], dtype=float)

    @staticmethod
    def _compute_metrics(
        daily_ret: pd.Series,
        excess: pd.Series,
        turnover_list: list[float],
        n_pos_list: list[int],
    ) -> dict[str, Any]:
        if len(daily_ret) == 0:
            return {}
        mean_d = float(daily_ret.mean())
        std_d = float(daily_ret.std(ddof=0))
        ex_mean = float(excess.mean())
        ex_std = float(excess.std(ddof=0))

        total_return = float((1 + daily_ret).prod() - 1)
        n_days = len(daily_ret)
        annual_return = (1 + mean_d) ** 252 - 1 if mean_d > -1 else -1.0
        volatility = std_d * np.sqrt(252)
        sharpe = (mean_d * 252) / (volatility + 1e-12)
        annual_excess = (1 + ex_mean) ** 252 - 1 if ex_mean > -1 else -1.0
        info_ratio = (ex_mean * 252) / (ex_std * np.sqrt(252) + 1e-12)

        equity = (1 + daily_ret).cumprod()
        peak = equity.cummax()
        dd = (equity / peak - 1)
        max_dd = float(dd.min()) if len(dd) > 0 else 0.0
        calmar = (annual_return / abs(max_dd)) if max_dd < 0 else float("inf")

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_excess": annual_excess,
            "volatility": volatility,
            "sharpe": sharpe,
            "info_ratio": info_ratio,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "avg_turnover": float(np.mean(turnover_list)) if turnover_list else 0.0,
            "win_rate": float((daily_ret > 0).mean()),
            "avg_positions": float(np.mean(n_pos_list)) if n_pos_list else 0.0,
            "n_days": n_days,
        }
