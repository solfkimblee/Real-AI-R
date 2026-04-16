"""HMM 制度识别测试。"""

from __future__ import annotations

import numpy as np

from real_ai_r.v9.regime.hmm import GaussianHMM, RegimeDetector


def _two_state_data(seed: int = 0, n_per: int = 80) -> np.ndarray:
    """生成双状态数据：前半段均值 [0, 0]，后半段均值 [3, 3]。"""
    rng = np.random.default_rng(seed)
    s1 = rng.normal(0, 0.5, size=(n_per, 2))
    s2 = rng.normal(3, 0.5, size=(n_per, 2))
    return np.vstack([s1, s2])


def test_hmm_fits_two_states_correctly() -> None:
    X = _two_state_data()
    hmm = GaussianHMM(n_states=2, n_iter=100, random_state=0).fit(X)
    assert hmm.start_prob_ is not None
    assert hmm.means_.shape == (2, 2)
    assert hmm.trans_mat_.shape == (2, 2)
    # 均值应接近 {0, 3}
    centers = np.sort(hmm.means_.mean(axis=1))
    assert centers[0] < 1.5
    assert centers[1] > 1.5


def test_hmm_predict_proba_shape() -> None:
    X = _two_state_data()
    hmm = GaussianHMM(n_states=2, n_iter=50, random_state=0).fit(X)
    p = hmm.predict_proba(X)
    assert p.shape == (X.shape[0], 2)
    # 每行和为 1
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_hmm_posterior_shifts_with_state() -> None:
    X = _two_state_data()
    hmm = GaussianHMM(n_states=2, n_iter=100, random_state=0).fit(X)
    p = hmm.predict_proba(X)
    # 前半段和后半段的多数状态应该不同
    first_arg = int(p[:80].mean(axis=0).argmax())
    last_arg = int(p[80:].mean(axis=0).argmax())
    assert first_arg != last_arg


def test_hmm_score_finite() -> None:
    X = _two_state_data()
    hmm = GaussianHMM(n_states=2, n_iter=30, random_state=0).fit(X)
    s = hmm.score(X)
    assert np.isfinite(s)


def test_hmm_raises_when_T_too_small() -> None:
    import pytest
    hmm = GaussianHMM(n_states=3)
    with pytest.raises(ValueError):
        hmm.fit(np.zeros((2, 4)))


def test_regime_detector_warmup_returns_uniform() -> None:
    det = RegimeDetector(n_states=3, min_train=50)
    # 数据不足
    feat = np.zeros((10, 2))
    p = det.infer(feat)
    assert p.shape == (3,)
    assert np.allclose(p, 1 / 3, atol=1e-9)


def test_regime_detector_trains_and_retrains() -> None:
    X = _two_state_data()
    det = RegimeDetector(
        n_states=2, min_train=60, retrain_every=20, random_state=0,
    )
    p1 = det.infer(X[:100])
    assert p1.shape == (2,)
    assert det._hmm is not None
    # 再次 infer：应该不重训
    means_before = det._hmm.means_.copy()
    det.infer(X[:110])
    assert np.allclose(det._hmm.means_, means_before)


def test_regime_detector_reset() -> None:
    X = _two_state_data()
    det = RegimeDetector(n_states=2, min_train=60, random_state=0)
    det.infer(X[:100])
    assert det._hmm is not None
    det.reset()
    assert det._hmm is None
    assert det._last_proba is None


def test_predict_last_is_consistent() -> None:
    X = _two_state_data()
    hmm = GaussianHMM(n_states=2, n_iter=50, random_state=0).fit(X)
    p = hmm.predict_proba(X)
    p_last = hmm.predict_last(X)
    assert np.allclose(p[-1], p_last)
