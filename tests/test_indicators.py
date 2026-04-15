"""单元测试 — 技术指标"""

from __future__ import annotations

import numpy as np
import pandas as pd

from real_ai_r.data.indicators import (
    add_all_indicators,
    add_bollinger,
    add_ema,
    add_ma,
    add_macd,
    add_rsi,
)


def _make_sample_data(n: int = 100) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    close = 10 + np.cumsum(rng.randn(n) * 0.3)
    return pd.DataFrame({"close": close})


class TestIndicators:
    def test_add_ma(self) -> None:
        df = _make_sample_data()
        df = add_ma(df, window=10)
        assert "ma_10" in df.columns
        # 前 9 个值应该是 NaN
        assert df["ma_10"].isna().sum() == 9

    def test_add_ema(self) -> None:
        df = _make_sample_data()
        df = add_ema(df, window=10)
        assert "ema_10" in df.columns

    def test_add_macd(self) -> None:
        df = _make_sample_data()
        df = add_macd(df)
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_hist" in df.columns

    def test_add_bollinger(self) -> None:
        df = _make_sample_data()
        df = add_bollinger(df)
        assert "bb_upper" in df.columns
        assert "bb_middle" in df.columns
        assert "bb_lower" in df.columns
        # 上轨 > 中轨 > 下轨
        valid = df.dropna()
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()

    def test_add_rsi(self) -> None:
        df = _make_sample_data()
        df = add_rsi(df, window=14)
        assert "rsi_14" in df.columns
        valid = df["rsi_14"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_add_all_indicators(self) -> None:
        df = _make_sample_data(n=200)
        df = add_all_indicators(df)
        expected_cols = ["ma_5", "ma_10", "ma_20", "ma_60", "macd", "bb_upper", "rsi_14"]
        for col in expected_cols:
            assert col in df.columns
