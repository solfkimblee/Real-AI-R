"""V9 状态管理测试。"""

from __future__ import annotations

import numpy as np

from real_ai_r.v9.state import V9State


def test_bounded_portfolio_history() -> None:
    s = V9State(portfolio_return_maxlen=5)
    for i in range(20):
        s.record_portfolio_excess(float(i))
    assert len(s.portfolio_excess_history) == 5
    assert list(s.portfolio_excess_history) == [15.0, 16.0, 17.0, 18.0, 19.0]


def test_bounded_factor_ic() -> None:
    s = V9State(factor_history_maxlen=3)
    for i in range(10):
        s.record_factor_ic("momentum", float(i))
    assert len(s.factor_ic_history["momentum"]) == 3
    assert list(s.factor_ic_history["momentum"]) == [7.0, 8.0, 9.0]


def test_board_return_matrix_shape() -> None:
    s = V9State()
    for i in range(10):
        s.record_board_return("A", float(i))
        s.record_board_return("B", float(-i))
    mat = s.get_board_return_matrix(["A", "B"], lookback=5)
    assert mat.shape == (5, 2)
    # 最后一行应是最新的 9, -9
    assert mat[-1, 0] == 9.0
    assert mat[-1, 1] == -9.0


def test_reset_clears_all() -> None:
    s = V9State()
    s.record_factor_ic("f", 0.1)
    s.record_board_return("A", 1.0)
    s.last_weights = {"A": 0.5}
    s.reset()
    assert len(s.factor_ic_history) == 0
    assert len(s.board_return_history) == 0
    assert s.last_weights == {}


def test_regime_features_maxlen() -> None:
    s = V9State(regime_feature_maxlen=4)
    for i in range(10):
        s.record_regime_feature(np.array([i, i + 1]))
    mat = s.get_regime_feature_matrix()
    assert mat.shape == (4, 2)
    assert mat[-1, 0] == 9


def test_serialization_roundtrip() -> None:
    s = V9State()
    s.record_factor_ic("mom", 0.1)
    s.record_factor_ic("mom", 0.2)
    s.record_board_return("A", 1.0)
    s.record_portfolio_excess(0.5)
    s.record_regime_feature(np.array([1.0, 2.0]))
    s.last_weights = {"A": 0.3}

    d = s.to_dict()
    s2 = V9State.from_dict(d)
    assert list(s2.factor_ic_history["mom"]) == [0.1, 0.2]
    assert list(s2.board_return_history["A"]) == [1.0]
    assert list(s2.portfolio_excess_history) == [0.5]
    assert s2.last_weights == {"A": 0.3}
