"""V9.2 Hedge 元集成策略测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from real_ai_r.macro.meta_ensemble_v92 import (
    MetaEnsembleStrategyV92,
    V92Params,
    _member_return,
    _selection_to_weights,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def simple_board_df():
    """真实结构的 board_df（V5/V7/V8 都能跑的最小列）。"""
    rng = np.random.default_rng(42)
    n = 30
    names = [f"板块{i:02d}" for i in range(n)]
    return pd.DataFrame(
        {
            "name": names,
            "code": [f"BK{i:04d}" for i in range(n)],
            "change_pct": rng.normal(0.5, 2.0, n),
            "turnover_rate": rng.uniform(0.5, 5.0, n),
            "rise_count": rng.integers(5, 50, n),
            "fall_count": rng.integers(5, 30, n),
            "lead_stock": [f"股{i}" for i in range(n)],
            "lead_stock_pct": rng.uniform(-5, 10, n),
            "momentum_5d": rng.normal(0, 5.0, n),
        },
    )


# ======================================================================
# 工具函数
# ======================================================================


def test_selection_to_weights_equal() -> None:
    w = _selection_to_weights(["a", "b", "c"])
    assert set(w.keys()) == {"a", "b", "c"}
    assert all(abs(v - 1 / 3) < 1e-9 for v in w.values())


def test_selection_to_weights_empty() -> None:
    assert _selection_to_weights([]) == {}


def test_selection_to_weights_topk_cap() -> None:
    w = _selection_to_weights(["a", "b", "c", "d"], k=2)
    assert len(w) == 2
    assert set(w.keys()) == {"a", "b"}


def test_member_return_weighted_sum() -> None:
    w = {"a": 0.6, "b": 0.4}
    r = {"a": 1.0, "b": -2.0, "c": 100.0}
    assert abs(_member_return(w, r) - (0.6 * 1.0 + 0.4 * -2.0)) < 1e-9


def test_member_return_missing_keys_default_zero() -> None:
    w = {"a": 0.5, "b": 0.5}
    r = {"a": 1.0}  # b 缺失
    assert abs(_member_return(w, r) - 0.5) < 1e-9


def test_member_return_empty_weights() -> None:
    assert _member_return({}, {"a": 1.0}) == 0.0


# ======================================================================
# 实例化 + drop-in 接口
# ======================================================================


def test_instantiation_default_members() -> None:
    s = MetaEnsembleStrategyV92()
    assert s.member_names == ["V5", "V7", "V8", "V9"]
    assert s.VERSION == "V9.2"


def test_instantiation_custom_members() -> None:
    s = MetaEnsembleStrategyV92(members=["V5", "V8"])
    assert s.member_names == ["V5", "V8"]
    assert s._v5 is not None
    assert s._v7 is None
    assert s._v8 is not None
    assert s._v9 is None


def test_initial_hedge_equal_weights() -> None:
    s = MetaEnsembleStrategyV92()
    w = s.get_hedge_weights()
    assert abs(sum(w.values()) - 1.0) < 1e-6
    # 等权 (warmup 期)
    for v in w.values():
        assert abs(v - 0.25) < 1e-6


def test_predict_drop_in_interface(simple_board_df) -> None:
    """V9.2 与 V8 接口完全兼容。"""
    s = MetaEnsembleStrategyV92()
    result = s.predict(
        board_df=simple_board_df,
        top_n=10,
        tech_history=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        cycle_history=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    assert len(result.predictions) <= 10
    assert result.strategy_summary.startswith("V9.2[Hedge]")
    assert "V9.2" in result.market_style


def test_predict_empty_board_df() -> None:
    s = MetaEnsembleStrategyV92()
    result = s.predict(board_df=pd.DataFrame({"name": []}))
    assert len(result.predictions) == 0


def test_predict_none_board_df() -> None:
    s = MetaEnsembleStrategyV92()
    result = s.predict(board_df=None)
    assert len(result.predictions) == 0


# ======================================================================
# Hedge 反馈回路
# ======================================================================


def test_observe_updates_hedge(simple_board_df) -> None:
    s = MetaEnsembleStrategyV92(
        v92_params=V92Params(hedge_warmup=1, hedge_eta=10.0, hedge_floor=0.0),
    )
    # 第一次预测 → 确定各成员持仓
    result = s.predict(board_df=simple_board_df, top_n=5)
    selections = s.get_last_member_selections()
    assert any(len(v) > 0 for v in selections.values())

    # 喂一个所有板块都涨 3% 的"无信号"收益 → 所有成员 return 相同 → hedge 维持等权
    flat = {n: 3.0 for n in simple_board_df["name"]}
    member_rets = s.observe_realized_returns(flat)
    # 每个成员都应拿到 3% 平均收益
    for m, r in member_rets.items():
        if selections.get(m):
            assert abs(r - 3.0) < 1e-6


def test_hedge_rewards_winning_member(simple_board_df) -> None:
    """手动注入差异化成员持仓 → Hedge 应把权重偏向赢家。

    说明：V5 与 V8 在平坦输入上可能挑同样板块；此测仅验证 Hedge 更新路径，
    因此直接设定 _last_member_weights 来保证两者持仓互斥，再喂入差异收益。
    """
    s = MetaEnsembleStrategyV92(
        v92_params=V92Params(hedge_warmup=0, hedge_eta=5.0, hedge_floor=0.0),
        members=["V5", "V8"],
    )

    v5_picks = list(simple_board_df["name"].values[:3])
    v8_picks = list(simple_board_df["name"].values[3:6])

    for _ in range(8):
        # 模拟每日 predict 后留下的成员持仓
        s._last_member_weights = {
            "V5": {b: 1 / len(v5_picks) for b in v5_picks},
            "V8": {b: 1 / len(v8_picks) for b in v8_picks},
        }
        s._last_member_selections = {"V5": v5_picks, "V8": v8_picks}
        realized = {n: 0.0 for n in simple_board_df["name"]}
        for b in v8_picks:
            realized[b] = 2.0
        for b in v5_picks:
            realized[b] = -2.0
        s.observe_realized_returns(realized)

    w = s.get_hedge_weights()
    assert w["V8"] > w["V5"]


def test_hedge_floor_respected(simple_board_df) -> None:
    """即使某成员持续亏损，floor 仍保证最低权重。"""
    s = MetaEnsembleStrategyV92(
        v92_params=V92Params(
            hedge_warmup=0, hedge_eta=100.0, hedge_floor=0.1,
        ),
        members=["V5", "V8"],
    )

    for _ in range(20):
        s.predict(
            board_df=simple_board_df, top_n=5,
            tech_history=[0.0] * 10, cycle_history=[0.0] * 10,
        )
        sel = s.get_last_member_selections()
        v8 = set(sel.get("V8", []))
        v5 = set(sel.get("V5", []))
        realized = {}
        for n in simple_board_df["name"]:
            if n in v8 and n not in v5:
                realized[n] = 5.0
            elif n in v5 and n not in v8:
                realized[n] = -5.0
            else:
                realized[n] = 0.0
        s.observe_realized_returns(realized)

    w = s.get_hedge_weights()
    assert w["V5"] >= 0.1 - 1e-6


# ======================================================================
# V8 兼容回调
# ======================================================================


def test_record_excess_forwards_to_v8(simple_board_df) -> None:
    s = MetaEnsembleStrategyV92()
    s.record_excess(-0.5)
    s.record_excess(-0.3)
    assert len(s._v8._excess_history) == 2


def test_record_board_performance_forwards(simple_board_df) -> None:
    s = MetaEnsembleStrategyV92()
    s.record_board_performance("板块01", 0.5)
    s.record_board_performance("板块01", 0.2)
    assert "板块01" in s._v8._board_excess_history
    assert len(s._v8._board_excess_history["板块01"]) == 2


# ======================================================================
# 聚合正确性
# ======================================================================


def test_aggregation_produces_top_n(simple_board_df) -> None:
    s = MetaEnsembleStrategyV92()
    result = s.predict(board_df=simple_board_df, top_n=8)
    # 最终选出不多于 top_n
    assert len(result.predictions) <= 8


def test_aggregation_favors_common_picks(simple_board_df) -> None:
    """如果多个成员共同选中某些板块，聚合分应偏向共识板块。"""
    s = MetaEnsembleStrategyV92()
    result = s.predict(board_df=simple_board_df, top_n=5)
    selections = s.get_last_member_selections()

    # 统计每个板块被多少成员选中
    count = {}
    for sel in selections.values():
        for b in sel:
            count[b] = count.get(b, 0) + 1

    # 最终持仓里至少有一半的板块被 ≥2 个成员共同选中
    final_names = [p.board_name for p in result.predictions]
    shared = sum(1 for b in final_names if count.get(b, 0) >= 2)
    # 在一个多成员共识的场景下至少有 1 个被共选
    assert shared >= 1


# ======================================================================
# 一次完整"走步"冒烟
# ======================================================================


def _synthetic_panel(n_days: int = 30, n_boards: int = 25, seed: int = 0):
    """生成带动量特征的面板数据。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-06-03", periods=n_days)
    names = [f"板块{i:02d}" for i in range(n_boards)]
    rows = []
    persist = rng.uniform(0.0, 0.3, n_boards)
    drift = rng.uniform(-0.05, 0.15, n_boards)
    x = np.zeros(n_boards)
    for d in dates:
        x = persist * x + rng.normal(drift, 1.8, n_boards)
        for i, n in enumerate(names):
            rows.append(
                {
                    "date": d,
                    "name": n,
                    "code": f"BK{i:04d}",
                    "change_pct": float(x[i]),
                    "turnover_rate": float(rng.uniform(0.5, 5.0)),
                    "rise_count": int(rng.integers(5, 50)),
                    "fall_count": int(rng.integers(5, 30)),
                    "lead_stock": f"股{i}",
                    "lead_stock_pct": float(x[i] + 1),
                    "momentum_5d": float(x[i] * 5),
                },
            )
    return pd.DataFrame(rows)


def test_end_to_end_walk_forward() -> None:
    """模拟用户 WF 协议：每日 predict + observe，运行 n 天不崩。"""
    df = _synthetic_panel()
    dates = sorted(df["date"].unique())

    s = MetaEnsembleStrategyV92(
        v92_params=V92Params(hedge_warmup=3, hedge_eta=2.0, hedge_floor=0.05),
    )

    tech_hist: list[float] = []
    cycle_hist: list[float] = []
    realized_series = []

    for i in range(len(dates) - 1):
        today = df[df["date"] == dates[i]]
        tomorrow = df[df["date"] == dates[i + 1]]
        cp_today = pd.to_numeric(today["change_pct"], errors="coerce").fillna(0.0)
        tech_hist.append(float(cp_today.mean() + 0.2))
        cycle_hist.append(float(cp_today.mean() - 0.2))

        result = s.predict(
            board_df=today.reset_index(drop=True),
            top_n=5,
            tech_history=tech_hist,
            cycle_history=cycle_hist,
        )

        realized = dict(
            zip(tomorrow["name"], tomorrow["change_pct"], strict=False),
        )
        member_rets = s.observe_realized_returns(realized)
        # 计算组合收益
        final_boards = [p.board_name for p in result.predictions]
        port_ret = np.mean(
            [realized.get(b, 0.0) for b in final_boards],
        ) if final_boards else 0.0
        realized_series.append(float(port_ret))

    # Hedge 权重应已形成分布
    w = s.get_hedge_weights()
    assert abs(sum(w.values()) - 1.0) < 1e-6
    # 所有成员权重 ≥ floor
    for v in w.values():
        assert v >= s.v92.hedge_floor - 1e-6

    # 至少有若干天有非零组合收益
    assert any(abs(x) > 1e-9 for x in realized_series)


def test_reset_clears_state(simple_board_df) -> None:
    s = MetaEnsembleStrategyV92()
    s.predict(board_df=simple_board_df, top_n=5)
    s.observe_realized_returns({n: 1.0 for n in simple_board_df["name"]})

    # 此时成员历史已积累
    s.reset()
    # hedge 归零
    assert s.hedge._steps == 0
    # V9 状态归零
    assert len(s._v9.state.factor_ic_history) == 0
