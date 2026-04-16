"""V9.3 Warm-Start 策略测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from real_ai_r.macro.v9_3_warmstart import V93Params, V9_3Strategy


# ======================================================================
# Fixtures
# ======================================================================


def _panel(n_days: int, n_boards: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    names = [f"板块{i:02d}" for i in range(n_boards)]
    persist = rng.uniform(0.0, 0.3, n_boards)
    drift = rng.uniform(-0.05, 0.15, n_boards)
    x = np.zeros(n_boards)
    rows = []
    for d in dates:
        x = persist * x + rng.normal(drift, 1.8)
        for i, n in enumerate(names):
            # lead_stock_pct: 每板块有独立的随机分量，避免 lead - change 恒定
            lead_delta = float(rng.normal(1.0, 2.0))
            rows.append(
                {
                    "date": d,
                    "name": n,
                    "code": f"BK{i:04d}",
                    "change_pct": float(x[i]),
                    "turnover_rate": float(rng.uniform(0.5, 5.0)),
                    "rise_count": int(rng.integers(5, 50)),
                    "fall_count": int(rng.integers(5, 30)),
                    "lead_stock_pct": float(x[i]) + lead_delta,
                    "momentum_5d": float(x[i] * 5),
                },
            )
    return pd.DataFrame(rows)


@pytest.fixture
def long_history():
    """120 天历史，足够训练 HMM。"""
    return _panel(120)


@pytest.fixture
def short_history():
    """20 天历史，不足以训 HMM（仅测 warmup 不崩）。"""
    return _panel(20)


@pytest.fixture
def single_day():
    return _panel(1).iloc[:20]


# ======================================================================
# 实例化与默认配置
# ======================================================================


def test_instantiation_default_factors() -> None:
    s = V9_3Strategy()
    # 默认 8 个因子：5 price + 3 breadth
    assert len(s.active_factors()) == 8
    assert "momentum_1d" in s.active_factors()
    assert "main_inflow" not in s.active_factors()  # 稀疏数据友好，不含 fund_flow
    assert not s._warmed


def test_v9_config_passes_sparse_friendly_defaults() -> None:
    s = V9_3Strategy()
    cfg = s.engine.config
    assert cfg.enable_graph is False  # 小样本下关图传播
    # 经扫参后: ra/shrinkage 取更平衡值（不过度防守）
    assert 0.5 < cfg.risk_aversion < 3.0
    assert 0.2 <= cfg.cov_shrinkage < 0.6


# ======================================================================
# Warmup
# ======================================================================


def test_fit_warmup_basic(long_history) -> None:
    s = V9_3Strategy()
    diag = s.fit_warmup(long_history)
    assert s._warmed
    assert diag["n_days"] > 60
    # 未被剪枝的因子应积累大量 IC
    active_ic = {
        f: diag["factor_ic_samples"].get(f, 0)
        for f in s.active_factors()
    }
    assert len(active_ic) >= 4
    # 大多数未剪因子累积 >= 60 条 IC
    assert sum(1 for v in active_ic.values() if v > 60) >= 3


def test_fit_warmup_trains_hmm(long_history) -> None:
    s = V9_3Strategy()
    diag = s.fit_warmup(long_history)
    assert diag["hmm_trained"] is True
    assert s.engine.regime._hmm is not None
    assert s.engine.regime._hmm.means_ is not None


def test_fit_warmup_short_history_no_hmm(short_history) -> None:
    s = V9_3Strategy()
    diag = s.fit_warmup(short_history)
    # 历史不足：HMM 未训
    assert diag["hmm_trained"] is False
    # 但 IC 仍应有积累
    assert diag["n_days"] >= 10


def test_fit_warmup_missing_columns_raises() -> None:
    df = pd.DataFrame({"name": ["a"], "x": [1]})
    s = V9_3Strategy()
    with pytest.raises(ValueError):
        s.fit_warmup(df)


def test_fit_warmup_empty_raises() -> None:
    s = V9_3Strategy()
    with pytest.raises(ValueError):
        s.fit_warmup(pd.DataFrame())


# ======================================================================
# 因子剪枝
# ======================================================================


def test_auto_prune_zero_factor() -> None:
    """构造一个因子恒等于 0 的场景 → 剪枝应生效。"""
    s = V9_3Strategy(
        v93_params=V93Params(
            prune_std_threshold=1e-3,
            prune_consecutive_days=3,
            prune_grace_period=2,
        ),
    )

    # 用较小数据启动 warmup（避免训 HMM 太慢）
    df = _panel(n_days=20)
    # 加一个"永远为 0"的 catalyst_score（event 因子会自动为 0，因为没有 extra_signals）
    # 把 event 因子也加进去
    custom_factors = list(V93Params().default_factor_names) + ["catalyst_score"]
    import dataclasses
    cfg = dataclasses.replace(
        s.engine.config, factor_names=custom_factors,
    )
    s.engine.config = cfg
    diag = s.fit_warmup(df)
    # catalyst_score 应被剪
    assert "catalyst_score" in s.disabled_factors()
    assert "catalyst_score" not in s.active_factors()


def test_no_prune_when_factor_has_signal(long_history) -> None:
    s = V9_3Strategy()
    s.fit_warmup(long_history)
    # momentum_1d 在真实数据上 std 不可能是 0
    assert "momentum_1d" not in s.disabled_factors()


# ======================================================================
# Predict — V8 drop-in 接口
# ======================================================================


def test_predict_before_warmup_works(single_day) -> None:
    s = V9_3Strategy()
    # 冷启动也能 predict，只是 IC 会退化为等权
    result = s.predict(board_df=single_day, top_n=5)
    assert len(result.predictions) <= 5


def test_predict_empty_df_returns_empty() -> None:
    s = V9_3Strategy()
    result = s.predict(board_df=pd.DataFrame({"name": []}))
    assert len(result.predictions) == 0


def test_predict_after_warmup_has_regime(long_history) -> None:
    s = V9_3Strategy()
    s.fit_warmup(long_history)
    last_day = long_history[long_history["date"] == long_history["date"].max()]
    result = s.predict(board_df=last_day.reset_index(drop=True), top_n=5)
    # 制度信息应注入 market_style
    assert "regime=" in result.market_style


def test_predict_top_n_respected(long_history) -> None:
    s = V9_3Strategy()
    s.fit_warmup(long_history)
    last_day = long_history[long_history["date"] == long_history["date"].max()]
    result = s.predict(board_df=last_day.reset_index(drop=True), top_n=3)
    assert len(result.predictions) <= 3


def test_predict_ignores_v8_specific_args(long_history) -> None:
    """V9.3 忽略 tech_history / cycle_history / fund_df 不应报错。"""
    s = V9_3Strategy()
    s.fit_warmup(long_history)
    last_day = long_history[long_history["date"] == long_history["date"].max()]
    result = s.predict(
        board_df=last_day.reset_index(drop=True),
        fund_df=pd.DataFrame({"whatever": [1]}),
        top_n=5,
        tech_history=[0.1, 0.2],
        cycle_history=[0.0, 0.1],
    )
    assert len(result.predictions) <= 5


# ======================================================================
# 状态持久化 — 跨窗口模拟
# ======================================================================


def test_state_persists_across_predicts(long_history) -> None:
    """多次 predict + observe 不 reset → IC 继续累积。"""
    s = V9_3Strategy()
    s.fit_warmup(long_history)
    ic_before = {
        k: len(v) for k, v in s.engine.state.factor_ic_history.items()
    }

    # 继续喂 5 天"新数据"
    extra = _panel(n_days=6, seed=99)
    dates = sorted(extra["date"].unique())
    for i, d in enumerate(dates[:-1]):
        today = extra[extra["date"] == d].reset_index(drop=True)
        tomorrow = extra[extra["date"] == dates[i + 1]]
        s.predict(board_df=today, top_n=5)
        realized = dict(
            zip(tomorrow["name"], tomorrow["change_pct"], strict=False),
        )
        s.observe_realized_returns(realized)

    ic_after = {
        k: len(v) for k, v in s.engine.state.factor_ic_history.items()
    }
    # IC 历史应变长
    for k in ic_before:
        assert ic_after.get(k, 0) >= ic_before[k]


def test_reset_clears_everything(long_history) -> None:
    s = V9_3Strategy()
    s.fit_warmup(long_history)
    assert s._warmed
    s.reset()
    assert not s._warmed
    assert len(s._disabled_factors) == 0
    assert len(s.engine.state.factor_ic_history) == 0


# ======================================================================
# 反馈接口 & V8 兼容回调
# ======================================================================


def test_observe_updates_ic(single_day) -> None:
    s = V9_3Strategy()
    s.predict(board_df=single_day, top_n=5)
    # 喂一组收益 → IC 应被记录
    realized = {n: float(np.random.randn()) for n in single_day["name"]}
    s.observe_realized_returns(realized)
    # 至少部分因子有 1 条 IC 记录
    assert any(
        len(v) > 0 for v in s.engine.state.factor_ic_history.values()
    )


def test_record_excess_noop() -> None:
    """V8 兼容占位: record_excess 不应报错且不做事。"""
    s = V9_3Strategy()
    s.record_excess(-0.5)
    s.record_board_performance("板块01", 0.1)
    # 没有可断言的副作用，但不应报错


# ======================================================================
# End-to-end: warmup → WF 模拟
# ======================================================================


def test_end_to_end_walk_forward_stateful() -> None:
    """warmup 60 天历史 + 继续 40 天 WF → 状态持续累积不报错。"""
    hist = _panel(n_days=60, seed=1)
    wf = _panel(n_days=40, seed=2)

    s = V9_3Strategy()
    diag = s.fit_warmup(hist)
    assert diag["n_days"] > 30

    dates = sorted(wf["date"].unique())
    for i in range(len(dates) - 1):
        today = wf[wf["date"] == dates[i]].reset_index(drop=True)
        tomorrow = wf[wf["date"] == dates[i + 1]]
        r = s.predict(board_df=today, top_n=5)
        realized = dict(
            zip(tomorrow["name"], tomorrow["change_pct"], strict=False),
        )
        s.observe_realized_returns(realized)
        assert r is not None

    # 活跃因子应累积 warmup + WF 的 IC (被剪因子不记录)
    active = s.active_factors()
    active_ic_counts = {
        f: len(s.engine.state.factor_ic_history.get(f, [])) for f in active
    }
    # 至少部分活跃因子累积 > 50 条
    assert sum(1 for v in active_ic_counts.values() if v > 50) >= 3
