"""V9 引擎集成测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_ai_r.v9.engine import V9Config, V9Engine


def test_engine_empty_df_returns_empty_prediction() -> None:
    eng = V9Engine()
    pred = eng.predict(board_df=pd.DataFrame({"name": []}))
    assert pred.n_positions == 0
    assert len(pred.weights) == 0


def test_engine_cold_start_equal_weights() -> None:
    """冷启动（IC 历史不足）时应该退化为等权因子。"""
    df = pd.DataFrame(
        {
            "name": [f"b{i}" for i in range(5)],
            "change_pct": np.arange(5),
            "turnover_rate": np.ones(5),
            "rise_count": [10] * 5,
            "fall_count": [10] * 5,
            "lead_stock_pct": np.arange(5),
        },
    )
    cfg = V9Config(
        enable_regime=False,
        enable_graph=False,
        max_positions=3,
    )
    eng = V9Engine(cfg)
    pred = eng.predict(df)
    assert pred.n_positions <= 3
    assert abs(pred.weights.sum() - 1.0) < 1e-3


def test_engine_predict_with_regime(panel_df) -> None:
    cfg = V9Config(
        enable_regime=True,
        enable_graph=False,
        regime_min_train=20,
        regime_retrain_every=5,
        max_positions=5,
    )
    eng = V9Engine(cfg)
    dates = sorted(panel_df["date"].unique())
    for d in dates[:40]:
        slice_df = panel_df[panel_df["date"] == d].reset_index(drop=True)
        feat = np.array(
            [
                float(slice_df["change_pct"].mean()),
                float(slice_df["change_pct"].std(ddof=0)),
            ],
        )
        pred = eng.predict(board_df=slice_df, regime_features=feat)
        # 喂下一期收益
        if dates.index(d) < len(dates) - 1:
            next_d = dates[dates.index(d) + 1]
            next_df = panel_df[panel_df["date"] == next_d]
            realized = dict(
                zip(next_df["name"], next_df["change_pct"], strict=False),
            )
            eng.update_feedback(realized_returns=realized)

    # 最终一次预测应有制度后验
    assert pred.regime_posterior is not None
    assert len(pred.regime_posterior) == cfg.regime_n_states


def test_engine_ic_history_grows(panel_df) -> None:
    eng = V9Engine(V9Config(enable_regime=False, enable_graph=False))
    dates = sorted(panel_df["date"].unique())
    for d in dates[:30]:
        slice_df = panel_df[panel_df["date"] == d].reset_index(drop=True)
        eng.predict(board_df=slice_df)
        if dates.index(d) < len(dates) - 1:
            next_d = dates[dates.index(d) + 1]
            next_df = panel_df[panel_df["date"] == next_d]
            realized = dict(
                zip(next_df["name"], next_df["change_pct"], strict=False),
            )
            eng.update_feedback(realized_returns=realized)
    # 一些因子的 IC 历史应该被累积
    assert len(eng.state.factor_ic_history) > 0
    # 每个因子都有若干 IC 记录
    any_factor = next(iter(eng.state.factor_ic_history.keys()))
    assert len(eng.state.factor_ic_history[any_factor]) > 0


def test_engine_respects_forbidden(panel_df) -> None:
    cfg = V9Config(
        enable_regime=False, enable_graph=False, max_positions=3,
    )
    eng = V9Engine(cfg)
    first_day = sorted(panel_df["date"].unique())[0]
    slice_df = panel_df[panel_df["date"] == first_day].reset_index(drop=True)
    forbidden = list(slice_df["name"].values[:3])
    pred = eng.predict(board_df=slice_df, forbidden=forbidden)
    for name in forbidden:
        assert pred.weights.get(name, 0.0) < 1e-4


def test_engine_diagnostics_has_expected_keys(panel_df) -> None:
    eng = V9Engine(V9Config(enable_regime=False, enable_graph=False))
    first_day = sorted(panel_df["date"].unique())[0]
    slice_df = panel_df[panel_df["date"] == first_day].reset_index(drop=True)
    pred = eng.predict(board_df=slice_df)
    for k in ("n_factors", "top_factor_weights", "portfolio_status"):
        assert k in pred.diagnostics


def test_engine_reset() -> None:
    eng = V9Engine()
    eng.state.record_factor_ic("f", 0.1)
    eng.state.last_weights = {"A": 0.5}
    eng.reset()
    assert len(eng.state.factor_ic_history) == 0
    assert eng.state.last_weights == {}


def test_engine_update_feedback_tracks_excess(panel_df) -> None:
    eng = V9Engine(V9Config(enable_regime=False, enable_graph=False))
    dates = sorted(panel_df["date"].unique())
    slice_df = panel_df[panel_df["date"] == dates[0]].reset_index(drop=True)
    eng.predict(slice_df)
    next_df = panel_df[panel_df["date"] == dates[1]]
    realized = dict(zip(next_df["name"], next_df["change_pct"], strict=False))
    result = eng.update_feedback(
        realized_returns=realized, benchmark_return=0.5,
    )
    assert "portfolio_return" in result
    assert "excess" in result
    assert len(eng.state.portfolio_excess_history) == 1
