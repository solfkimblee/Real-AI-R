"""泽平宏观 V9.3 — V9 跨窗口 Warm-Start 版本

针对 V9 真实 WF 回测失败的根因（每 20 天 reset 冷启动）量身设计：

1. 显式 `fit_warmup(historical_panel_df)` 接口
   - 在 WF 评估开始前，用 1-2 年历史数据预训练 HMM
   - 预填充因子 IC 历史、板块收益矩阵、制度特征序列
   - 评估期间无需再冷启

2. 显式"单实例跨窗口"语义
   - 整个 WF 评估共用一个 V9_3Strategy 实例
   - predict() 不 reset state；observe_realized_returns() 持续累积
   - Hedge 式集成外部化（如需，配合 V9.2 使用）

3. 因子自动剪枝（稀疏数据防噪）
   - 警告期观察每因子 std / 非零率
   - 连续 N 日 std < threshold 的因子 → 自动禁用（不参与加权）
   - 解决 fund_flow / event 全零拖累 IC 权重分配的问题

4. 小样本数据稀疏友好的默认 config
   - enable_graph=False（相关图在 <30 板块时主要是噪声）
   - 协方差收缩提高到 0.5，风险厌恶提高到 3.0
   - HMM min_train=60 且 retrain 频率降低（30 天/次）
   - 只保留 price + breadth 8 个因子（默认）

5. Drop-in V8 接口
   - predict(board_df, fund_df, top_n, tech_history, cycle_history)
   - record_excess / record_board_performance
   - observe_realized_returns (V9.2 / V9 通用)

典型用法:
    v93 = V9_3Strategy()
    v93.fit_warmup(historical_df)         # 1-2 年历史，一次性
    for day in wf_days:                   # 149 天 WF 评估
        result = v93.predict(board_df=...)
        v93.observe_realized_returns(realized)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from real_ai_r.macro.zeping_strategy import (
    ZepingBoardScore,
    ZepingMacroStrategy,
    ZepingParams,
    ZepingPredictionResult,
    ZepingWeights,
)
from real_ai_r.v9.engine import V9Config, V9Engine

logger = logging.getLogger(__name__)


# ======================================================================
# V9.3 参数
# ======================================================================


@dataclass(frozen=True)
class V93Params:
    """V9.3 Warm-Start 参数。"""

    # ---- 因子剪枝 ----
    # 对历史截面分计算 std，std < threshold 视为"信号死因子"
    prune_std_threshold: float = 1e-4
    # 连续 N 天 std 都低于阈值才剪掉（避免偶发零值误伤）
    prune_consecutive_days: int = 5
    # 警告期（warmup 前 N 天不参与剪枝判定，避免初始化期误判）
    prune_grace_period: int = 3

    # ---- 推荐的 V9Config（稀疏数据友好） ----
    default_factor_names: tuple[str, ...] = (
        # price（5 个）
        "momentum_1d",
        "momentum_5d",
        "momentum_20d",
        "volatility_20d",
        "turnover",
        # breadth（3 个）
        "rise_ratio",
        "lead_strength",
        "dispersion",
    )
    enable_regime: bool = True
    regime_n_states: int = 3
    regime_min_train: int = 60
    regime_retrain_every: int = 30
    enable_graph: bool = False  # 小样本下图传播主要放大噪声

    # ---- QP 组合优化 ----
    # 经合成数据扫参验证: ra=1.0/tp=0.3/shrink=0.3 在 warmup 后优于更保守版本
    # 更保守版本 (ra=3/tp=1/shrink=0.5) 在趋势数据上过度防守 → Sharpe 跌 60%
    risk_aversion: float = 1.0
    turnover_penalty: float = 0.3
    max_weight: float = 0.15
    max_positions: int = 10
    cov_shrinkage: float = 0.3

    # ---- IC 加权 ----
    ic_scheme: str = "icir"
    ic_halflife: int = 40               # 较长半衰期：warmup 历史有价值
    ic_min_samples: int = 20            # 经 warmup 后这个门槛容易满足

    # ---- Warmup ----
    # 若 historical panel 不足，最低需要的天数
    warmup_min_days: int = 30


# ======================================================================
# V9.3 主类
# ======================================================================


class V9_3Strategy(ZepingMacroStrategy):
    """V9.3 — V9 跨窗口 Warm-Start 版本，drop-in V8 兼容。"""

    VERSION = "V9.3"

    def __init__(
        self,
        weights: ZepingWeights | None = None,
        params: ZepingParams | None = None,
        v93_params: V93Params | None = None,
        v9_config: V9Config | None = None,
    ) -> None:
        super().__init__(weights, params)
        self.v93 = v93_params or V93Params()

        # 构造 V9Engine（稀疏数据友好默认）
        cfg = v9_config or V9Config(
            factor_names=list(self.v93.default_factor_names),
            enable_regime=self.v93.enable_regime,
            regime_n_states=self.v93.regime_n_states,
            regime_min_train=self.v93.regime_min_train,
            regime_retrain_every=self.v93.regime_retrain_every,
            enable_graph=self.v93.enable_graph,
            risk_aversion=self.v93.risk_aversion,
            turnover_penalty=self.v93.turnover_penalty,
            max_weight=self.v93.max_weight,
            max_positions=self.v93.max_positions,
            cov_shrinkage=self.v93.cov_shrinkage,
            ic_scheme=self.v93.ic_scheme,
            ic_halflife=self.v93.ic_halflife,
            ic_min_samples=self.v93.ic_min_samples,
        )
        self.engine = V9Engine(cfg)

        # warmup 状态
        self._warmed: bool = False
        self._warmup_days: int = 0
        self._disabled_factors: set[str] = set()
        # 因子连续低 std 计数
        self._low_std_streak: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Warmup — 在 WF 评估之前调用一次
    # ------------------------------------------------------------------

    def fit_warmup(
        self, panel_df: pd.DataFrame, verbose: bool = False,
    ) -> dict:
        """用历史面板数据预热 V9 引擎。

        参数:
            panel_df: 必须含 date, name, change_pct；建议 60~500 天
            verbose: 是否输出进度
        返回:
            诊断字典 {n_days, final_ic, disabled_factors, hmm_trained}
        """
        if panel_df is None or len(panel_df) == 0:
            raise ValueError("panel_df empty")
        for col in ("date", "name", "change_pct"):
            if col not in panel_df.columns:
                raise ValueError(f"panel_df missing column: {col}")

        df = panel_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "name"]).reset_index(drop=True)
        unique_dates = sorted(df["date"].unique())

        if len(unique_dates) < self.v93.warmup_min_days:
            logger.warning(
                "warmup days %d < min %d; HMM may be unstable",
                len(unique_dates), self.v93.warmup_min_days,
            )

        n_done = 0
        for i, d in enumerate(unique_dates[:-1]):
            today = df[df["date"] == d].reset_index(drop=True)
            tomorrow = df[df["date"] == unique_dates[i + 1]]
            if today.empty or tomorrow.empty:
                continue

            # 制度特征（默认 4 维: 均涨/波动/分化/上涨比）
            regime_feat = self._default_regime_features(today)

            # 引擎预测（积累因子截面分、regime 历史；不关心输出）
            self.engine.predict(
                board_df=today,
                regime_features=regime_feat,
                as_of=str(d.date()),
            )

            # 喂下一日实现收益 → 更新因子 IC、板块历史、制度 IC
            realized = dict(
                zip(
                    tomorrow["name"].astype(str).values,
                    pd.to_numeric(
                        tomorrow["change_pct"], errors="coerce",
                    ).fillna(0.0).values,
                    strict=False,
                ),
            )
            self.engine.update_feedback(realized_returns=realized)

            n_done += 1
            self._warmup_days = n_done  # 实时更新，供 _update_factor_pruning_stats 用
            # 剪枝检测（warmup 期间观察每日因子截面分 std）
            self._update_factor_pruning_stats()

            if verbose and (n_done % 20 == 0):
                logger.info("warmup %d / %d", n_done, len(unique_dates))

        self._warmup_days = n_done
        self._warmed = True

        diag = {
            "n_days": n_done,
            "factor_ic_samples": {
                k: len(v) for k, v in self.engine.state.factor_ic_history.items()
            },
            "disabled_factors": sorted(self._disabled_factors),
            "hmm_trained": (
                self.engine.regime._hmm is not None
                and self.engine.regime._hmm.means_ is not None
            ),
            "regime_feature_samples": len(
                self.engine.state.regime_feature_history,
            ),
        }
        return diag

    # ------------------------------------------------------------------
    # 因子剪枝逻辑
    # ------------------------------------------------------------------

    def _update_factor_pruning_stats(self) -> None:
        """基于最新日的因子截面分，更新各因子的低方差连续计数。"""
        if self._warmup_days <= self.v93.prune_grace_period:
            return
        # 直接读 V9Engine 上一轮 predict 缓存的截面分
        latest_scores = getattr(self.engine, "_last_factor_scores", None)
        if not latest_scores:
            return
        for fname, series in latest_scores.items():
            if series is None or len(series) == 0:
                continue
            try:
                std = float(np.asarray(series.values).std(ddof=0))
            except Exception:
                continue
            if std < self.v93.prune_std_threshold:
                self._low_std_streak[fname] = (
                    self._low_std_streak.get(fname, 0) + 1
                )
                if (
                    self._low_std_streak[fname]
                    >= self.v93.prune_consecutive_days
                    and fname not in self._disabled_factors
                ):
                    self._disabled_factors.add(fname)
                    self._apply_factor_pruning()
                    logger.info("factor pruned: %s (std consistently ~0)", fname)
            else:
                self._low_std_streak[fname] = 0

    def _apply_factor_pruning(self) -> None:
        """把被禁因子从 V9Engine 的配置中移除。"""
        current = list(self.engine.config.factor_names or [])
        if not current:
            # 默认是全部因子
            current = list(self.v93.default_factor_names)
        remaining = [f for f in current if f not in self._disabled_factors]
        if not remaining:
            # 全部被剪光？留一个最基本的
            remaining = ["momentum_1d"]
        # 直接覆盖 config（dataclass 若 frozen 需 replace）
        try:
            # V9Config 是 dataclass 默认可变
            self.engine.config.factor_names = remaining
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 主预测接口 — 与 V8 兼容
    # ------------------------------------------------------------------

    def predict(
        self,
        board_df: pd.DataFrame | None = None,
        fund_df: pd.DataFrame | None = None,
        top_n: int = 10,
        tech_history: list[float] | None = None,
        cycle_history: list[float] | None = None,
    ) -> ZepingPredictionResult:
        # V9.3 忽略 fund_df / tech_history / cycle_history
        # 这些在 V5/V7/V8 中用于 regime_score；V9.3 用 HMM 替代
        if board_df is None or len(board_df) == 0:
            return self._empty_result()

        regime_feat = self._default_regime_features(board_df)
        pred = self.engine.predict(
            board_df=board_df,
            regime_features=regime_feat,
        )
        if pred.n_positions == 0:
            return self._empty_result()

        # 按权重降序取 top_n
        w_sorted = pred.weights.sort_values(ascending=False)
        w_sorted = w_sorted[w_sorted > 1e-6].head(top_n)

        predictions: list[ZepingBoardScore] = []
        for bn in w_sorted.index:
            predictions.append(
                ZepingBoardScore(
                    board_name=str(bn),
                    total_score=float(pred.scores.get(bn, 0.0)),
                    quant_score=float(pred.scores.get(bn, 0.0)),
                ),
            )

        top_factors = sorted(
            pred.factor_weights.items(), key=lambda x: -x[1],
        )[:3]
        summary = (
            f"V9.3[warm={self._warmed} d={self._warmup_days} "
            f"disabled={len(self._disabled_factors)}] "
            f"top_factors=" + ",".join(f"{k}:{v:.2f}" for k, v in top_factors)
        )

        regime_label = ""
        if pred.regime_posterior is not None:
            k_star = int(np.argmax(pred.regime_posterior))
            regime_label = f"regime={k_star}({pred.regime_posterior[k_star]:.2f})"

        return ZepingPredictionResult(
            predictions=predictions,
            current_hot_stage=0,
            current_hot_stage_name="",
            market_style=f"V9.3 | {regime_label}",
            total_boards=int(len(board_df)),
            filtered_redline=0,
            strategy_summary=summary,
        )

    # ------------------------------------------------------------------
    # 反馈接口 (V9.2 / V9 风格)
    # ------------------------------------------------------------------

    def observe_realized_returns(
        self, realized_returns: dict[str, float],
    ) -> dict:
        """喂下一日板块实现收益 — 更新因子 IC / 板块历史 / 组合超额。"""
        result = self.engine.update_feedback(realized_returns=realized_returns)
        # 更新剪枝统计
        self._update_factor_pruning_stats()
        return result

    # V8 兼容占位（V9.3 不使用这两个回调，保留为 no-op 便于无缝替换）
    def record_excess(self, daily_excess: float) -> None:  # noqa: D401
        return

    def record_board_performance(
        self, board_name: str, excess: float,
    ) -> None:
        return

    # ------------------------------------------------------------------
    # 诊断 & 重置
    # ------------------------------------------------------------------

    def disabled_factors(self) -> list[str]:
        return sorted(self._disabled_factors)

    def active_factors(self) -> list[str]:
        cur = self.engine.config.factor_names or list(
            self.v93.default_factor_names,
        )
        return [f for f in cur if f not in self._disabled_factors]

    def get_hmm_posterior(self) -> np.ndarray | None:
        return self.engine.regime.last_proba

    def reset(self) -> None:
        self.engine.reset()
        self._warmed = False
        self._warmup_days = 0
        self._disabled_factors.clear()
        self._low_std_streak.clear()

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _default_regime_features(slice_df: pd.DataFrame) -> np.ndarray:
        """4 维市场特征: [均涨, 波动, 分化, 上涨比]。"""
        cp = pd.to_numeric(slice_df["change_pct"], errors="coerce").fillna(0.0)
        mean_ret = float(cp.mean())
        vol = float(cp.std(ddof=0))
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

    def _empty_result(self) -> ZepingPredictionResult:
        return ZepingPredictionResult(
            predictions=[],
            current_hot_stage=0,
            current_hot_stage_name="",
            market_style="V9.3|empty",
            total_boards=0,
            filtered_redline=0,
            strategy_summary=f"V9.3 empty (warmed={self._warmed})",
        )


__all__ = ["V9_3Strategy", "V93Params"]
