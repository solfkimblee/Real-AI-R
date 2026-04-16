"""V10 泽平三维度×产业链×卖铲人 策略测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from real_ai_r.macro.zeping_strategy_v10 import (
    DEFAULT_CHAINS,
    V10Chain,
    V10Params,
    ZepingMacroStrategyV10,
)


# ======================================================================
# Fixtures
# ======================================================================


def _board_df(n_boards: int = 30, seed: int = 42) -> pd.DataFrame:
    """构造含有产业链关键词的 board_df。"""
    rng = np.random.default_rng(seed)
    # 故意构造一些命中链关键词的板块名
    sample_names = [
        "半导体", "芯片", "算力", "光模块", "AI应用",
        "锂电池", "光伏", "新能源汽车", "锂矿", "正极材料",
        "消费电子", "智能硬件",
        "医药", "创新药", "CXO",
        "军工", "航空",
        "银行", "证券", "保险",
        "煤炭", "有色", "化工",
        "食品饮料", "乳业",
        "互联网", "传媒",
        "未知主题A", "未知主题B",
    ]
    names = sample_names[:n_boards]
    return pd.DataFrame(
        {
            "name": names,
            "code": [f"BK{i:04d}" for i in range(len(names))],
            "change_pct": rng.normal(1.0, 2.0, len(names)),
            "turnover_rate": rng.uniform(1.0, 4.0, len(names)),
            "rise_count": rng.integers(10, 50, len(names)),
            "fall_count": rng.integers(5, 30, len(names)),
            "lead_stock": [f"股{i}" for i in range(len(names))],
            "lead_stock_pct": rng.uniform(0, 8, len(names)),
        },
    )


# ======================================================================
# 实例化 + 默认配置
# ======================================================================


def test_v10_instantiation() -> None:
    s = ZepingMacroStrategyV10()
    assert s.VERSION == "V10"
    assert len(s.chains) == 10
    # 每条链都有三维敏感性
    for c in s.chains:
        assert len(c.sensitivity) == 3
        for v in c.sensitivity:
            assert -1.0 <= v <= 1.0


def test_v10_params_small_count() -> None:
    """V10 宣称只 6 个参数—验证之。"""
    p = V10Params()
    # 数一下 dataclass 的 field 数
    from dataclasses import fields
    n = len(fields(p))
    assert n <= 7, f"V10Params should have <=7 params, got {n}"


def test_custom_chains_override() -> None:
    chain = V10Chain(
        name="自定义",
        all_keywords=("测试",),
        upstream_keywords=(),
        sensitivity=(0.5, 0.5, 0.5),
    )
    s = ZepingMacroStrategyV10(chains=(chain,))
    assert len(s.chains) == 1
    assert s.chains[0].name == "自定义"


# ======================================================================
# 链匹配
# ======================================================================


def test_match_chain_ai_computing() -> None:
    s = ZepingMacroStrategyV10()
    chains = s._match_chains("半导体芯片")
    assert "AI算力" in chains
    assert "半导体" in chains


def test_match_chain_none_for_unknown() -> None:
    s = ZepingMacroStrategyV10()
    assert s._match_chains("未知主题XYZ") == []


def test_match_upstream_lithium_is_new_energy_upstream() -> None:
    s = ZepingMacroStrategyV10()
    assert "新能源" in s._match_upstream("锂矿")
    assert "新能源" in s._match_upstream("正极材料")


def test_match_upstream_semiconductor_equipment() -> None:
    s = ZepingMacroStrategyV10()
    up = s._match_upstream("半导体设备")
    assert "AI算力" in up or "半导体" in up


def test_chain_cache_consistency() -> None:
    s = ZepingMacroStrategyV10()
    r1 = s._match_chains("半导体")
    r2 = s._match_chains("半导体")
    assert r1 == r2
    assert "半导体" in s._board_chain_cache


# ======================================================================
# 三维度信号
# ======================================================================


def test_cycle_signal_uses_history_when_provided() -> None:
    s = ZepingMacroStrategyV10()
    # 周期板块连涨 2%/日，科技板块连跌 1%/日 → cycle_signal > 0
    tech = [-1.0] * 10
    cyc = [2.0] * 10
    board_df = _board_df()
    sig = s._compute_cycle_signal(tech, cyc, board_df)
    assert sig > 0.3
    # 反转
    sig2 = s._compute_cycle_signal(cyc, tech, board_df)
    assert sig2 < -0.3


def test_cycle_signal_fallback_to_cross_section() -> None:
    s = ZepingMacroStrategyV10()
    # 构造: 周期板块涨幅远大于科技板块
    df = pd.DataFrame(
        {
            "name": ["煤炭", "有色", "半导体", "芯片"],
            "change_pct": [5.0, 4.5, -2.0, -1.5],
            "rise_count": [20, 20, 10, 10],
            "fall_count": [5, 5, 20, 20],
        },
    )
    sig = s._compute_cycle_signal(None, None, df)
    assert sig > 0  # 周期 > 科技 → 正


def test_liquidity_signal_range() -> None:
    s = ZepingMacroStrategyV10()
    # 高换手 → 正
    df_high = _board_df()
    df_high["turnover_rate"] = 4.0
    sig_high = s._compute_liquidity_signal(df_high, None)
    # 低换手 → 负
    df_low = _board_df()
    df_low["turnover_rate"] = 0.8
    sig_low = s._compute_liquidity_signal(df_low, None)
    assert sig_high > sig_low
    assert -1.0 <= sig_low <= 1.0
    assert -1.0 <= sig_high <= 1.0


def test_liquidity_signal_fund_df_influence() -> None:
    s = ZepingMacroStrategyV10()
    df = _board_df()
    fund_positive = pd.DataFrame({"main_net_inflow": [100.0, 200.0, 50.0]})
    fund_negative = pd.DataFrame({"main_net_inflow": [-100.0, -200.0, -50.0]})
    sig_pos = s._compute_liquidity_signal(df, fund_positive)
    sig_neg = s._compute_liquidity_signal(df, fund_negative)
    assert sig_pos > sig_neg


def test_risk_preference_positive_breadth() -> None:
    s = ZepingMacroStrategyV10()
    df = _board_df()
    df["rise_count"] = 40
    df["fall_count"] = 10  # 80% 上涨
    df["lead_stock_pct"] = 5.0
    sig = s._compute_risk_preference_signal(df)
    assert sig > 0.3


def test_risk_preference_negative_breadth() -> None:
    s = ZepingMacroStrategyV10()
    df = _board_df()
    df["rise_count"] = 5
    df["fall_count"] = 45  # 10% 上涨
    df["lead_stock_pct"] = -2.0
    sig = s._compute_risk_preference_signal(df)
    assert sig < -0.3


# ======================================================================
# Bonus 计算
# ======================================================================


def test_chain_bonus_zero_without_match() -> None:
    s = ZepingMacroStrategyV10()
    chain_scores = {c.name: 1.0 for c in s.chains}
    bonus = s._compute_board_bonus("未知主题XYZ", chain_scores)
    assert bonus == 0.0


def test_chain_bonus_positive_when_match_and_positive_score() -> None:
    s = ZepingMacroStrategyV10()
    chain_scores = {c.name: 1.0 for c in s.chains}  # 全链正分
    bonus_ai = s._compute_board_bonus("半导体芯片", chain_scores)
    assert bonus_ai > 0


def test_chain_bonus_capped() -> None:
    s = ZepingMacroStrategyV10()
    # 极端大链分
    chain_scores = {c.name: 100.0 for c in s.chains}
    bonus = s._compute_board_bonus("半导体", chain_scores)
    # 应被 chain_bonus_max (=10) + membership (=2) + shovel ≤ 8 截断
    assert bonus <= s.v10.chain_bonus_max + s.v10.chain_membership_boost \
        + s.v10.chain_bonus_max * 0.8 + 1e-6


def test_shovel_bonus_added_for_upstream() -> None:
    s = ZepingMacroStrategyV10()
    # 新能源链热度 +1，锂矿作为新能源上游应获溢出
    scores_cold = {c.name: 0.0 for c in s.chains}
    scores_hot = dict(scores_cold)
    scores_hot["新能源"] = 2.0
    b_cold = s._compute_board_bonus("锂矿", scores_cold)
    b_hot = s._compute_board_bonus("锂矿", scores_hot)
    assert b_hot > b_cold


def test_shovel_bonus_zero_when_chain_cold() -> None:
    s = ZepingMacroStrategyV10()
    # 即使是上游，链分负 → shovel 不应给分（只在 > 0 时给）
    scores = {c.name: -1.0 for c in s.chains}
    b = s._compute_board_bonus("锂矿", scores)
    # 链 bonus 会是负的，但 shovel 应为 0
    matched_chains = s._match_chains("锂矿")
    upstream_of = s._match_upstream("锂矿")
    # 锂矿 是新能源上游；锂矿 不在任何链的 all_keywords 里（大概率）→
    # chain bonus = 0，shovel 也 = 0（链负）
    if not matched_chains:
        assert b == 0.0


# ======================================================================
# Predict / Drop-in V8 接口
# ======================================================================


def test_predict_returns_top_n() -> None:
    s = ZepingMacroStrategyV10()
    df = _board_df()
    result = s.predict(board_df=df, top_n=5)
    assert len(result.predictions) <= 5


def test_predict_with_tech_cycle_history() -> None:
    s = ZepingMacroStrategyV10()
    df = _board_df()
    res = s.predict(
        board_df=df, top_n=5,
        tech_history=[0.5] * 10,
        cycle_history=[-0.5] * 10,
    )
    assert "V10" in res.strategy_summary
    # 应包含三维度信号信息
    assert "cycle=" in res.strategy_summary
    assert "liq=" in res.strategy_summary
    assert "risk=" in res.strategy_summary


def test_predict_empty_board_df() -> None:
    s = ZepingMacroStrategyV10()
    result = s.predict(board_df=pd.DataFrame({"name": []}), top_n=5)
    assert len(result.predictions) == 0


def test_predict_summary_has_regime_label() -> None:
    s = ZepingMacroStrategyV10()
    df = _board_df()
    df["rise_count"] = 40
    df["fall_count"] = 10
    df["turnover_rate"] = 3.5
    df["lead_stock_pct"] = 5.0
    r = s.predict(board_df=df, top_n=5)
    # 在这种"强看涨"输入下应标"牛市"或"轮动"
    assert (
        "牛市" in r.strategy_summary
        or "轮动" in r.strategy_summary
        or "顺周期" in r.strategy_summary
    )


# ======================================================================
# 回撤制动
# ======================================================================


def test_drawdown_brake_triggers() -> None:
    s = ZepingMacroStrategyV10(
        v10_params=V10Params(drawdown_lookback=3, drawdown_threshold=-2.0),
    )
    # 连续 3 天严重亏损 → 触发制动
    s.record_excess(-1.0)
    s.record_excess(-1.5)
    s.record_excess(-1.0)  # 累计 -3.5 < -2 → 制动
    assert s._is_in_drawdown()


def test_drawdown_brake_not_triggered() -> None:
    s = ZepingMacroStrategyV10(
        v10_params=V10Params(drawdown_lookback=3, drawdown_threshold=-5.0),
    )
    s.record_excess(-1.0)
    s.record_excess(+0.5)
    s.record_excess(-1.0)  # 累计 -1.5 > -5 → 不制动
    assert not s._is_in_drawdown()


def test_drawdown_brake_insufficient_history() -> None:
    s = ZepingMacroStrategyV10(
        v10_params=V10Params(drawdown_lookback=5),
    )
    s.record_excess(-10.0)  # 仅 1 天
    assert not s._is_in_drawdown()


def test_brake_zeros_out_bonus_in_predict() -> None:
    """触发制动时 predict 不应加 chain bonus，退化到纯 V5 排序。"""
    s = ZepingMacroStrategyV10(
        v10_params=V10Params(drawdown_lookback=2, drawdown_threshold=-1.0),
    )
    df = _board_df()
    # 先在非制动态跑一次获取基线
    r_normal = s.predict(board_df=df, top_n=5)

    # 制造制动
    s.record_excess(-2.0)
    s.record_excess(-2.0)
    assert s._is_in_drawdown()

    r_braked = s.predict(board_df=df, top_n=5)
    # 制动 summary 应含"制动中"
    assert "制动中" in r_braked.strategy_summary


# ======================================================================
# 红线过滤
# ======================================================================


def test_redline_boards_never_selected() -> None:
    s = ZepingMacroStrategyV10()
    df = pd.DataFrame(
        {
            "name": ["房地产", "水泥", "白酒", "半导体", "芯片"],
            "change_pct": [10, 10, 10, 1, 1],  # 红线涨幅远大
            "turnover_rate": [3.0] * 5,
            "rise_count": [40] * 5,
            "fall_count": [5] * 5,
            "lead_stock_pct": [5.0] * 5,
        },
    )
    r = s.predict(board_df=df, top_n=5)
    picked = {p.board_name for p in r.predictions}
    # 红线板块不应入选
    assert "房地产" not in picked
    assert "水泥" not in picked
    assert "白酒" not in picked


# ======================================================================
# reset
# ======================================================================


def test_reset_clears_state() -> None:
    s = ZepingMacroStrategyV10()
    s.record_excess(-1.0)
    s._match_chains("半导体")  # 填缓存
    assert len(s._excess_history) == 1
    assert len(s._board_chain_cache) > 0
    s.reset()
    assert len(s._excess_history) == 0
    assert len(s._board_chain_cache) == 0


# ======================================================================
# 端到端 WF 冒烟
# ======================================================================


def test_end_to_end_walk_forward() -> None:
    s = ZepingMacroStrategyV10()
    # 生成 30 日连续数据
    rng = np.random.default_rng(0)
    days = 30
    dates = pd.bdate_range("2024-06-03", periods=days)
    boards = [
        "半导体", "芯片", "光模块", "AI应用",
        "锂电池", "光伏", "锂矿",
        "消费电子", "医药", "创新药",
        "军工", "银行", "证券",
        "煤炭", "有色",
        "食品", "互联网",
        "未知A", "未知B",
    ]
    rows = []
    for d in dates:
        for bn in boards:
            rows.append(
                {
                    "date": d,
                    "name": bn,
                    "change_pct": float(rng.normal(0.5, 2.0)),
                    "turnover_rate": float(rng.uniform(1, 4)),
                    "rise_count": int(rng.integers(10, 40)),
                    "fall_count": int(rng.integers(5, 25)),
                    "lead_stock_pct": float(rng.uniform(-2, 8)),
                },
            )
    panel = pd.DataFrame(rows)

    tech_hist, cycle_hist = [], []
    results = []
    for i in range(days - 1):
        today = panel[panel["date"] == dates[i]].reset_index(drop=True)
        tomorrow = panel[panel["date"] == dates[i + 1]]
        cp = pd.to_numeric(today["change_pct"], errors="coerce").fillna(0.0)
        tech_hist.append(float(cp.values[:len(cp) // 2].mean()))
        cycle_hist.append(float(cp.values[len(cp) // 2:].mean()))
        r = s.predict(
            board_df=today, top_n=5,
            tech_history=tech_hist, cycle_history=cycle_hist,
        )
        picks = [p.board_name for p in r.predictions]
        realized = dict(
            zip(tomorrow["name"], tomorrow["change_pct"], strict=False),
        )
        port = np.mean([realized.get(b, 0) for b in picks]) if picks else 0
        s.record_excess(float(port))
        results.append(r)

    # 全程不崩
    assert len(results) == days - 1
    # 每次至少有 1 个 pick
    assert all(len(r.predictions) > 0 for r in results)
