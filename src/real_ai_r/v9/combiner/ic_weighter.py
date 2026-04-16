"""IC 动态因子加权器。

每日因子 IC = 因子分数与下一期板块收益的 rank 相关系数（Spearman）。
历史 IC 序列 → ICIR 加权 或 指数衰减加权 → 得到当期因子权重。

可选：按 HMM 后验 × 制度条件 IC 做软融合（制度感知加权）。

这是对 V8 静态 40/40/20 的根本性超越：
- 权重自适应市场环境，完全免调参
- 下个月某因子失效，IC 归零 → 权重自动降为零
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def rank_ic(scores: pd.Series, forward_returns: pd.Series) -> float:
    """计算 rank-IC（Spearman 相关）。"""
    if scores is None or forward_returns is None:
        return 0.0
    df = pd.DataFrame(
        {"s": scores, "r": forward_returns},
    ).dropna()
    if len(df) < 3:
        return 0.0
    r_s = df["s"].rank()
    r_r = df["r"].rank()
    if r_s.std(ddof=0) < 1e-9 or r_r.std(ddof=0) < 1e-9:
        return 0.0
    ic = np.corrcoef(r_s.values, r_r.values)[0, 1]
    if np.isnan(ic):
        return 0.0
    return float(ic)


@dataclass
class ICWeighter:
    """根据历史 IC 序列计算因子权重。

    权重方案：
    - "icir"     : w ∝ mean(IC) / std(IC)（风险调整 IC）
    - "ic_mean"  : w ∝ mean(IC)
    - "ewma_ic"  : 指数加权近期 IC
    - "equal"    : 等权（基线）

    参数:
        scheme: 上面四个之一
        halflife: ewma 半衰期
        min_samples: 最少 IC 样本数，低于则回退到 equal
        temperature: softmax 温度；温度越低权重越集中
        clip_negative: True 时负权重（无效因子）截断为 0
    """

    scheme: str = "icir"
    halflife: int = 20
    min_samples: int = 10
    temperature: float = 1.0
    clip_negative: bool = True

    def compute_weights(
        self, ic_history: dict[str, list[float]],
    ) -> dict[str, float]:
        """从 {factor: [ic_t...]} 计算当期因子权重（归一化到和为 1）。"""
        factor_names = list(ic_history.keys())
        if not factor_names:
            return {}

        enough = all(
            len(ic_history[f]) >= self.min_samples for f in factor_names
        )
        if not enough:
            # 回退: 等权
            w = 1.0 / len(factor_names)
            return {f: w for f in factor_names}

        scores = np.zeros(len(factor_names))

        if self.scheme == "equal":
            scores[:] = 1.0
        elif self.scheme == "ic_mean":
            for i, f in enumerate(factor_names):
                scores[i] = float(np.mean(ic_history[f]))
        elif self.scheme == "icir":
            for i, f in enumerate(factor_names):
                arr = np.array(ic_history[f], dtype=float)
                mu = arr.mean()
                sd = arr.std(ddof=0)
                scores[i] = mu / (sd + 1e-6)
        elif self.scheme == "ewma_ic":
            alpha = 1.0 - 0.5 ** (1.0 / max(self.halflife, 1))
            for i, f in enumerate(factor_names):
                arr = np.array(ic_history[f], dtype=float)
                w_t = np.array(
                    [(1 - alpha) ** (len(arr) - 1 - t) for t in range(len(arr))],
                )
                w_t /= w_t.sum() + 1e-12
                scores[i] = float(np.sum(w_t * arr))
        else:
            raise ValueError(f"unknown scheme: {self.scheme}")

        if self.clip_negative:
            scores = np.clip(scores, 0.0, None)

        total = scores.sum()
        if total < 1e-9:
            # 全部非正 → 等权
            w = 1.0 / len(factor_names)
            return {f: w for f in factor_names}

        # softmax 锐化 (可选)
        if self.temperature != 1.0 and self.temperature > 0:
            tempered = scores / max(self.temperature, 1e-6)
            tempered = tempered - tempered.max()
            e = np.exp(tempered)
            weights = e / (e.sum() + 1e-12)
        else:
            weights = scores / total

        return {factor_names[i]: float(weights[i]) for i in range(len(factor_names))}

    def compute_regime_weights(
        self,
        ic_history_by_regime: dict[int, dict[str, list[float]]],
        regime_posterior: np.ndarray,
    ) -> dict[str, float]:
        """制度条件 IC 加权 — 为每个 regime 计算一套权重，再按后验软融合。

        ic_history_by_regime: {regime_idx: {factor_name: [ic_t...]}}
        regime_posterior:   (K,) 当日制度后验
        """
        K = len(regime_posterior)
        all_factors: set[str] = set()
        for r in range(K):
            all_factors.update(ic_history_by_regime.get(r, {}).keys())
        factor_names = sorted(all_factors)
        if not factor_names:
            return {}

        merged = {f: 0.0 for f in factor_names}
        for r in range(K):
            w_r = self.compute_weights(
                ic_history_by_regime.get(r, {f: [] for f in factor_names}),
            )
            p_r = float(regime_posterior[r])
            for f in factor_names:
                merged[f] += p_r * w_r.get(f, 0.0)

        total = sum(merged.values())
        if total < 1e-9:
            w = 1.0 / len(factor_names)
            return {f: w for f in factor_names}
        return {f: v / total for f, v in merged.items()}

    @staticmethod
    def combine(
        factor_scores: dict[str, pd.Series],
        weights: dict[str, float],
    ) -> pd.Series:
        """按权重线性组合因子分数。"""
        if not factor_scores or not weights:
            return pd.Series(dtype=float)

        # 用第一个因子的 index 做基准
        first = next(iter(factor_scores.values()))
        if first is None or len(first) == 0:
            return pd.Series(dtype=float)
        index = first.index

        out = pd.Series(np.zeros(len(index)), index=index)
        for f, w in weights.items():
            s = factor_scores.get(f)
            if s is None or len(s) == 0:
                continue
            out = out.add(s.reindex(index).fillna(0.0) * w, fill_value=0.0)
        return out
