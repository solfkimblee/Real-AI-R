"""Gaussian HMM 隐制度识别 — 零外部依赖实现。

用 EM (Baum-Welch) 训练，forward 算法做在线推断。
输入 T×D 特征矩阵，学到 K 个制度的均值/协方差/转移矩阵，
推断时返回最后一日的后验概率 π(s|x_{1:T})。

相比 V8 的 regime_score（一维连续值、取 min 抹杀信号）：
- HMM 学到多维制度结构（如 [收益, 波动, 分化, 资金]）
- 后验是分布，支持"软融合"多套因子权重
- 转移矩阵捕捉 regime 持续性，天然平滑

依赖：numpy + scipy（logsumexp、multivariate_normal）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


@dataclass
class GaussianHMM:
    """离散隐状态 + 高斯观测的 HMM。

    参数:
        n_states: 隐制度数（K）
        n_iter: EM 最大迭代
        tol: log-lik 收敛阈值
        reg_covar: 协方差正则（避免奇异）
        random_state: 随机种子
    """

    n_states: int = 3
    n_iter: int = 50
    tol: float = 1e-3
    reg_covar: float = 1e-4
    random_state: int = 42

    # 学到的参数
    start_prob_: np.ndarray | None = None         # (K,)
    trans_mat_: np.ndarray | None = None          # (K, K)
    means_: np.ndarray | None = None              # (K, D)
    covars_: np.ndarray | None = None             # (K, D, D)
    converged_: bool = False
    n_features_: int = 0
    log_lik_history_: list[float] = field(default_factory=list)

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> GaussianHMM:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D (T, D)")
        T, D = X.shape
        K = self.n_states
        if T < K:
            raise ValueError(f"Need T>=n_states, got T={T}, K={K}")
        self.n_features_ = D

        rng = np.random.default_rng(self.random_state)

        # --- 初始化: 用 kmeans++ 风格选均值 ---
        idx = [rng.integers(T)]
        for _ in range(K - 1):
            dists = np.min(
                np.stack(
                    [
                        np.sum((X - X[i]) ** 2, axis=1) for i in idx
                    ],
                ),
                axis=0,
            )
            probs = dists / (dists.sum() + 1e-12)
            nxt = int(rng.choice(T, p=probs))
            idx.append(nxt)
        means = X[idx].copy()
        covars = np.stack(
            [
                np.cov(X.T) + np.eye(D) * self.reg_covar
                for _ in range(K)
            ],
        )
        start = np.ones(K) / K
        trans = np.full((K, K), 0.1 / (K - 1 if K > 1 else 1))
        np.fill_diagonal(trans, 0.9)
        trans = trans / trans.sum(axis=1, keepdims=True)

        prev_ll = -np.inf
        self.log_lik_history_ = []

        for it in range(self.n_iter):
            # --- E step: forward-backward ---
            log_emissions = self._log_emissions(X, means, covars)  # (T, K)
            log_alpha = self._forward(log_emissions, start, trans)  # (T, K)
            log_beta = self._backward(log_emissions, trans)          # (T, K)

            # Posterior gamma and xi
            log_gamma = log_alpha + log_beta
            log_gamma = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)  # (T, K)

            # xi: (T-1, K, K) = P(s_t=i, s_{t+1}=j | X)
            log_trans = np.log(trans + 1e-300)
            log_xi = (
                log_alpha[:-1, :, None]
                + log_trans[None, :, :]
                + log_emissions[1:, None, :]
                + log_beta[1:, None, :]
            )
            # normalize per t
            log_xi -= logsumexp(
                log_xi, axis=(1, 2), keepdims=True,
            )
            xi = np.exp(log_xi)

            # log-likelihood
            ll = float(logsumexp(log_alpha[-1]))
            self.log_lik_history_.append(ll)

            # --- M step ---
            start = gamma[0] / (gamma[0].sum() + 1e-12)
            trans = xi.sum(axis=0) / (
                xi.sum(axis=(0, 2), keepdims=True).squeeze(-1)
                + 1e-12
            )
            trans = trans / (trans.sum(axis=1, keepdims=True) + 1e-12)

            # means / covars
            gamma_sum = gamma.sum(axis=0) + 1e-12  # (K,)
            means = (gamma.T @ X) / gamma_sum[:, None]
            covars = np.zeros((K, D, D))
            for k in range(K):
                diff = X - means[k]
                covars[k] = (
                    (gamma[:, k][:, None] * diff).T @ diff
                ) / gamma_sum[k]
                covars[k] += np.eye(D) * self.reg_covar

            # convergence
            if it > 0 and abs(ll - prev_ll) < self.tol:
                self.converged_ = True
                break
            prev_ll = ll

        self.start_prob_ = start
        self.trans_mat_ = trans
        self.means_ = means
        self.covars_ = covars
        return self

    # ------------------------------------------------------------------
    # 推断
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回 (T, K) 后验概率，每行是 P(s_t | x_{1:t})（过滤分布）。"""
        if self.means_ is None:
            raise RuntimeError("HMM not fitted")
        X = np.asarray(X, dtype=float)
        log_em = self._log_emissions(X, self.means_, self.covars_)
        log_alpha = self._forward(log_em, self.start_prob_, self.trans_mat_)
        log_post = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
        return np.exp(log_post)

    def predict_last(self, X: np.ndarray) -> np.ndarray:
        """最后一个时间步的后验 (K,)。"""
        return self.predict_proba(X)[-1]

    def score(self, X: np.ndarray) -> float:
        """log-likelihood。"""
        X = np.asarray(X, dtype=float)
        log_em = self._log_emissions(X, self.means_, self.covars_)
        log_alpha = self._forward(log_em, self.start_prob_, self.trans_mat_)
        return float(logsumexp(log_alpha[-1]))

    # ------------------------------------------------------------------
    # 低层算法
    # ------------------------------------------------------------------

    @staticmethod
    def _log_emissions(
        X: np.ndarray, means: np.ndarray, covars: np.ndarray,
    ) -> np.ndarray:
        T = X.shape[0]
        K = means.shape[0]
        out = np.empty((T, K))
        for k in range(K):
            try:
                out[:, k] = multivariate_normal.logpdf(
                    X, mean=means[k], cov=covars[k], allow_singular=True,
                )
            except Exception:
                # 极端情况下回退到简化对角高斯
                diag = np.diag(covars[k]) + 1e-6
                diff = X - means[k]
                out[:, k] = -0.5 * (
                    np.sum(diff**2 / diag, axis=1)
                    + np.sum(np.log(2 * np.pi * diag))
                )
        return out

    @staticmethod
    def _forward(
        log_em: np.ndarray,
        start: np.ndarray,
        trans: np.ndarray,
    ) -> np.ndarray:
        T, K = log_em.shape
        log_alpha = np.zeros((T, K))
        log_alpha[0] = np.log(start + 1e-300) + log_em[0]
        log_trans = np.log(trans + 1e-300)
        for t in range(1, T):
            # log_alpha[t, j] = log_em[t, j] + logsumexp_i(log_alpha[t-1,i]+log_trans[i,j])
            log_alpha[t] = log_em[t] + logsumexp(
                log_alpha[t - 1, :, None] + log_trans, axis=0,
            )
        return log_alpha

    @staticmethod
    def _backward(
        log_em: np.ndarray, trans: np.ndarray,
    ) -> np.ndarray:
        T, K = log_em.shape
        log_beta = np.zeros((T, K))
        log_trans = np.log(trans + 1e-300)
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(
                log_trans + log_em[t + 1][None, :] + log_beta[t + 1][None, :],
                axis=1,
            )
        return log_beta


@dataclass
class RegimeDetector:
    """在线制度识别器封装。

    负责：
    - 累积市场特征历史
    - 周期性重训练 HMM（避免每日训）
    - 返回当日制度后验
    """

    n_states: int = 3
    min_train: int = 60
    retrain_every: int = 20
    n_iter: int = 50
    random_state: int = 42

    _hmm: GaussianHMM | None = None
    _since_train: int = 0
    _last_proba: np.ndarray | None = None

    def infer(self, feature_history: np.ndarray) -> np.ndarray:
        """给定 (T, D) 市场特征历史，返回当日 K 维制度后验。"""
        if feature_history is None or len(feature_history) < self.min_train:
            # 数据不足：均匀后验
            return np.ones(self.n_states) / self.n_states

        need_retrain = (
            self._hmm is None
            or self._since_train >= self.retrain_every
            or self._hmm.means_ is None
        )
        if need_retrain:
            self._hmm = GaussianHMM(
                n_states=self.n_states,
                n_iter=self.n_iter,
                random_state=self.random_state,
            ).fit(feature_history)
            self._since_train = 0
        else:
            self._since_train += 1

        proba = self._hmm.predict_last(feature_history)
        self._last_proba = proba
        return proba

    @property
    def last_proba(self) -> np.ndarray | None:
        return self._last_proba

    def reset(self) -> None:
        self._hmm = None
        self._since_train = 0
        self._last_proba = None
