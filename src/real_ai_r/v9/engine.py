"""V9 主引擎 — 编排所有组件，提供统一的 predict() 接口。

执行流程（日频）:

    board_df + context
         │
         ▼
    [1] 原子因子计算 → {factor_name: zscore Series}
         │
         ▼
    [2] HMM 制度识别（可选）→ 制度后验 π(s)
         │
         ▼
    [3] IC 动态加权（可按制度条件） → 因子权重 {f: w}
         │
         ▼
    [4] 线性组合 → 原始综合分 μ
         │
         ▼
    [5] 图传播（可选）→ 主题增强分 μ'
         │
         ▼
    [6] QP 组合优化 → 权重 w（Top-k + 换手成本 + 风险约束）
         │
         ▼
    [7] 输出 V9Prediction（含权重、分数、制度后验、诊断）

回测时调用 `update_feedback(next_returns)` 喂下一期板块收益，
系统内部更新：
- 因子 IC 历史（用于下轮 IC 加权）
- 组合实现收益 / 超额
- Hedge 集成成员权重（若启用）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from real_ai_r.v9.combiner.hedge_ensemble import HedgeEnsemble
from real_ai_r.v9.combiner.ic_weighter import ICWeighter, rank_ic
from real_ai_r.v9.factors import registry as factor_registry
from real_ai_r.v9.factors.base import FactorContext
from real_ai_r.v9.graph.propagation import GraphPropagator, build_correlation_graph
from real_ai_r.v9.optimizer.portfolio_qp import PortfolioOptimizer, PortfolioResult
from real_ai_r.v9.regime.hmm import RegimeDetector
from real_ai_r.v9.state import V9State


# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------


@dataclass
class V9Config:
    """V9 全局配置。"""

    # --- 因子选择 ---
    factor_names: list[str] | None = None  # None = 使用全部已注册因子

    # --- IC 加权 ---
    ic_scheme: str = "icir"               # equal / ic_mean / icir / ewma_ic
    ic_halflife: int = 20
    ic_min_samples: int = 10
    regime_conditional_ic: bool = True     # 用 HMM 后验加权分制度 IC

    # --- HMM 制度 ---
    enable_regime: bool = True
    regime_n_states: int = 3
    regime_min_train: int = 60
    regime_retrain_every: int = 20

    # --- 图传播 ---
    enable_graph: bool = True
    graph_knn: int = 10
    graph_threshold: float = 0.3
    graph_alpha: float = 0.25
    graph_steps: int = 2
    graph_corr_lookback: int = 60          # 构图回看
    graph_rebuild_every: int = 20          # 每 N 天重建

    # --- 组合优化 ---
    risk_aversion: float = 2.0
    turnover_penalty: float = 0.5
    max_weight: float = 0.15
    max_positions: int = 12
    cov_shrinkage: float = 0.2
    cov_lookback: int = 60

    # --- Hedge 集成 ---
    enable_hedge: bool = False
    hedge_members: list[str] = field(default_factory=list)
    hedge_eta: float = 2.0
    hedge_warmup: int = 10

    # --- 状态 ---
    board_return_maxlen: int = 120
    factor_history_maxlen: int = 120


# ----------------------------------------------------------------------
# 输出
# ----------------------------------------------------------------------


@dataclass
class V9Prediction:
    """单日 V9 预测结果。"""

    weights: pd.Series                      # index=board_name, 权重
    scores: pd.Series                       # 综合分
    factor_scores: dict[str, pd.Series]     # 各因子 zscore
    factor_weights: dict[str, float]        # 当日因子权重
    regime_posterior: np.ndarray | None      # (K,) 制度后验 or None
    top_boards: list[str]                   # 持仓板块名（权重 > 0）
    n_positions: int
    turnover: float
    portfolio_status: str
    diagnostics: dict[str, Any]


# ----------------------------------------------------------------------
# 主引擎
# ----------------------------------------------------------------------


class V9Engine:
    """V9 核心引擎。"""

    def __init__(
        self,
        config: V9Config | None = None,
        state: V9State | None = None,
    ) -> None:
        self.config = config or V9Config()
        self.state = state or V9State(
            factor_history_maxlen=self.config.factor_history_maxlen,
            board_return_maxlen=self.config.board_return_maxlen,
        )

        # 组件
        self.ic_weighter = ICWeighter(
            scheme=self.config.ic_scheme,
            halflife=self.config.ic_halflife,
            min_samples=self.config.ic_min_samples,
        )
        self.graph_prop = GraphPropagator(
            alpha=self.config.graph_alpha,
            n_steps=self.config.graph_steps,
        )
        self.regime = RegimeDetector(
            n_states=self.config.regime_n_states,
            min_train=self.config.regime_min_train,
            retrain_every=self.config.regime_retrain_every,
        )
        self.optimizer = PortfolioOptimizer(
            risk_aversion=self.config.risk_aversion,
            turnover_penalty=self.config.turnover_penalty,
            max_weight=self.config.max_weight,
            max_positions=self.config.max_positions,
            shrinkage=self.config.cov_shrinkage,
        )
        self.hedge: HedgeEnsemble | None = None
        if self.config.enable_hedge and self.config.hedge_members:
            self.hedge = HedgeEnsemble(
                members=list(self.config.hedge_members),
                eta=self.config.hedge_eta,
                warmup=self.config.hedge_warmup,
            )

        # 缓存
        self._graph_W: np.ndarray | None = None
        self._graph_names: list[str] = []
        self._graph_age: int = 10**9
        self._last_scores: pd.Series | None = None
        self._last_board_names: list[str] = []

        # 制度条件 IC 历史 { regime_idx: {factor: [ic_t]} }
        self._ic_by_regime: dict[int, dict[str, list[float]]] = {}

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------

    def predict(
        self,
        board_df: pd.DataFrame,
        extra_signals: dict[str, dict[str, float]] | None = None,
        regime_features: np.ndarray | None = None,
        forbidden: list[str] | None = None,
        as_of: str | None = None,
    ) -> V9Prediction:
        """执行一次日频预测。

        board_df: 截面 DataFrame，必须含 `name` 列
        extra_signals: 板块级附加信号 {board: {key: value}}（传给因子）
        regime_features: 当日市场特征向量 (D,) 或已累计的 (T, D)
            如传单日向量，内部会累计到 state.regime_feature_history。
            若 None 则跳过 HMM。
        forbidden: 红线板块（权重强制为 0）
        as_of: 当前日期字符串（用于诊断）
        """
        cfg = self.config
        if board_df is None or len(board_df) == 0:
            return self._empty_prediction()

        board_df = board_df.copy().reset_index(drop=True)
        if "name" not in board_df.columns:
            raise ValueError("board_df must contain 'name' column")

        # --- 积累市场特征 ---
        if regime_features is not None:
            feat = np.asarray(regime_features, dtype=float)
            if feat.ndim == 1:
                self.state.record_regime_feature(feat)

        # --- 1. 因子计算 ---
        factor_names = cfg.factor_names or factor_registry.names()
        # 构建 context
        board_names = list(board_df["name"].values)
        ret_matrix = self.state.get_board_return_matrix(
            board_names,
            lookback=min(60, self.state.board_return_maxlen),
        )
        # 若全为 NaN / 空，传 None
        if ret_matrix.size == 0 or np.isnan(ret_matrix).all():
            ret_matrix_for_ctx = None
        else:
            ret_matrix_for_ctx = ret_matrix

        ctx = FactorContext(
            board_return_matrix=ret_matrix_for_ctx,
            board_names=board_names,
            extra_signals=extra_signals or {},
            as_of=as_of,
        )
        factor_scores: dict[str, pd.Series] = {}
        for fname in factor_names:
            try:
                f = factor_registry.get(fname)
            except KeyError:
                continue
            try:
                s = f.compute(board_df, ctx)
                if s is not None and len(s) > 0:
                    # 对齐到 board_names，缺失填 0
                    s = s.reindex(board_names).fillna(0.0)
                    factor_scores[fname] = s
            except Exception:
                continue

        if not factor_scores:
            return self._empty_prediction()

        # --- 2. 制度识别 ---
        regime_post: np.ndarray | None = None
        if cfg.enable_regime:
            feat_hist = self.state.get_regime_feature_matrix()
            if feat_hist is not None and len(feat_hist) >= cfg.regime_min_train:
                regime_post = self.regime.infer(feat_hist)

        # --- 3. IC 动态加权 ---
        if cfg.regime_conditional_ic and regime_post is not None:
            weights_map = self.ic_weighter.compute_regime_weights(
                self._ic_by_regime, regime_post,
            )
            # 如果某因子不在 weights_map（从未记录），退化到全局权重
            if not weights_map or sum(weights_map.values()) < 1e-9:
                weights_map = self.ic_weighter.compute_weights(
                    self.state.factor_ic_history,
                )
        else:
            weights_map = self.ic_weighter.compute_weights(
                self.state.factor_ic_history,
            )

        # 只保留当前 factor_scores 存在的因子
        weights_map = {
            f: w for f, w in weights_map.items() if f in factor_scores
        }
        if not weights_map:
            # 完全冷启动 → 等权
            w0 = 1.0 / len(factor_scores)
            weights_map = {f: w0 for f in factor_scores}
        # 重归一化
        wsum = sum(weights_map.values()) or 1.0
        weights_map = {f: w / wsum for f, w in weights_map.items()}

        # --- 4. 线性组合 ---
        raw_scores = ICWeighter.combine(factor_scores, weights_map)

        # --- 5. 图传播 ---
        if cfg.enable_graph and ret_matrix_for_ctx is not None:
            if (
                self._graph_W is None
                or self._graph_age >= cfg.graph_rebuild_every
                or set(self._graph_names) != set(board_names)
            ):
                k = min(cfg.graph_corr_lookback, ret_matrix.shape[0])
                W, names = build_correlation_graph(
                    ret_matrix[-k:, :],
                    board_names,
                    knn=cfg.graph_knn,
                    threshold=cfg.graph_threshold,
                )
                self._graph_W = W
                self._graph_names = names
                self._graph_age = 0
            else:
                self._graph_age += 1
            scores = self.graph_prop.propagate(
                raw_scores, self._graph_W, self._graph_names,
            )
        else:
            scores = raw_scores

        # --- 6. QP 组合优化 ---
        cov_k = min(cfg.cov_lookback, ret_matrix.shape[0] if ret_matrix.size > 0 else 0)
        cov_matrix = None
        if cov_k >= 5:
            cov_matrix = ret_matrix[-cov_k:, :]

        result: PortfolioResult = self.optimizer.optimize(
            expected_returns=scores,
            return_matrix=cov_matrix,
            prev_weights=self.state.last_weights,
            forbidden=forbidden,
        )

        weights = result.weights
        # 更新上期权重
        self.state.last_weights = {
            n: float(w) for n, w in weights.items() if w > 1e-6
        }

        top_boards = [n for n, w in weights.items() if w > 1e-6]

        # --- 缓存本次分数，供下一轮 update_feedback 计算 IC ---
        self._last_scores = scores.copy()
        self._last_board_names = board_names
        self._last_factor_scores = factor_scores
        self._last_regime_post = regime_post

        diagnostics = {
            "n_factors": len(factor_scores),
            "regime_posterior": regime_post.tolist() if regime_post is not None else None,
            "top_factor_weights": sorted(
                weights_map.items(), key=lambda x: -x[1],
            )[:5],
            "portfolio_status": result.status,
            "objective": result.objective,
            "graph_edges": int((self._graph_W > 0).sum()) if self._graph_W is not None else 0,
        }

        return V9Prediction(
            weights=weights,
            scores=scores,
            factor_scores=factor_scores,
            factor_weights=weights_map,
            regime_posterior=regime_post,
            top_boards=top_boards,
            n_positions=result.n_positions,
            turnover=result.turnover,
            portfolio_status=result.status,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # 反馈（回测 / 实盘喂数据）
    # ------------------------------------------------------------------

    def update_feedback(
        self,
        realized_returns: dict[str, float],
        benchmark_return: float = 0.0,
        member_returns: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        喂入"下一期"板块实现收益，更新 IC、板块历史、组合超额、Hedge 权重。

        realized_returns: {board_name: next_period_return(%)}
        benchmark_return: 基准收益（如上证指数）
        member_returns: Hedge 集成各成员收益（可选）

        返回: 当期 IC 字典和组合实际超额
        """
        # 1. 板块历史收益
        for bn, r in realized_returns.items():
            self.state.record_board_return(bn, float(r))

        # 2. 因子 IC
        ic_dict: dict[str, float] = {}
        if self._last_scores is not None:
            forward = pd.Series(realized_returns)
            forward = forward.reindex(self._last_scores.index).astype(float)
            # 全局 IC
            for fname, fscore in getattr(self, "_last_factor_scores", {}).items():
                ic = rank_ic(fscore, forward)
                ic_dict[fname] = ic
                self.state.record_factor_ic(fname, ic)
            # 制度条件 IC
            if self._last_regime_post is not None:
                k_star = int(np.argmax(self._last_regime_post))
                by_regime = self._ic_by_regime.setdefault(k_star, {})
                for fname, ic in ic_dict.items():
                    by_regime.setdefault(fname, []).append(ic)
                    # 控制长度
                    if len(by_regime[fname]) > self.config.factor_history_maxlen:
                        by_regime[fname] = by_regime[fname][
                            -self.config.factor_history_maxlen:
                        ]

        # 3. 组合实际超额
        portfolio_ret = 0.0
        if self.state.last_weights:
            for bn, w in self.state.last_weights.items():
                portfolio_ret += w * float(realized_returns.get(bn, 0.0))
        excess = portfolio_ret - benchmark_return
        self.state.record_portfolio_excess(excess)

        # 4. Hedge 集成
        if self.hedge is not None and member_returns:
            self.hedge.update(member_returns)

        return {
            "ics": ic_dict,
            "portfolio_return": portfolio_ret,
            "excess": excess,
        }

    # ------------------------------------------------------------------
    # 其他
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.state.reset()
        self.regime.reset()
        if self.hedge is not None:
            self.hedge.reset()
        self._graph_W = None
        self._graph_names = []
        self._graph_age = 10**9
        self._last_scores = None
        self._ic_by_regime.clear()

    def _empty_prediction(self) -> V9Prediction:
        return V9Prediction(
            weights=pd.Series(dtype=float),
            scores=pd.Series(dtype=float),
            factor_scores={},
            factor_weights={},
            regime_posterior=None,
            top_boards=[],
            n_positions=0,
            turnover=0.0,
            portfolio_status="EMPTY",
            diagnostics={},
        )
