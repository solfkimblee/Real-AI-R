"""技术指标计算模块

基于 ta 库，提供常用技术指标的统一计算接口。
"""

from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands


def add_ma(df: pd.DataFrame, window: int = 20, column: str = "close") -> pd.DataFrame:
    """添加简单移动平均线（SMA）。"""
    col_name = f"ma_{window}"
    indicator = SMAIndicator(close=df[column], window=window)
    df[col_name] = indicator.sma_indicator()
    return df


def add_ema(df: pd.DataFrame, window: int = 20, column: str = "close") -> pd.DataFrame:
    """添加指数移动平均线（EMA）。"""
    col_name = f"ema_{window}"
    indicator = EMAIndicator(close=df[column], window=window)
    df[col_name] = indicator.ema_indicator()
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "close",
) -> pd.DataFrame:
    """添加 MACD 指标（macd, macd_signal, macd_hist）。"""
    indicator = MACD(close=df[column], window_fast=fast, window_slow=slow, window_sign=signal)
    df["macd"] = indicator.macd()
    df["macd_signal"] = indicator.macd_signal()
    df["macd_hist"] = indicator.macd_diff()
    return df


def add_bollinger(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
    column: str = "close",
) -> pd.DataFrame:
    """添加布林带指标（bb_upper, bb_middle, bb_lower）。"""
    indicator = BollingerBands(close=df[column], window=window, window_dev=num_std)
    df["bb_upper"] = indicator.bollinger_hband()
    df["bb_middle"] = indicator.bollinger_mavg()
    df["bb_lower"] = indicator.bollinger_lband()
    return df


def add_rsi(df: pd.DataFrame, window: int = 14, column: str = "close") -> pd.DataFrame:
    """添加 RSI 指标。"""
    indicator = RSIIndicator(close=df[column], window=window)
    df[f"rsi_{window}"] = indicator.rsi()
    return df


def add_stoch_rsi(df: pd.DataFrame, window: int = 14, column: str = "close") -> pd.DataFrame:
    """添加随机 RSI 指标。"""
    indicator = StochRSIIndicator(close=df[column], window=window)
    df["stoch_rsi"] = indicator.stochrsi()
    df["stoch_rsi_k"] = indicator.stochrsi_k()
    df["stoch_rsi_d"] = indicator.stochrsi_d()
    return df


def add_all_indicators(
    df: pd.DataFrame,
    ma_windows: list[int] | None = None,
) -> pd.DataFrame:
    """一次性添加所有常用指标。"""
    if ma_windows is None:
        ma_windows = [5, 10, 20, 60]

    for w in ma_windows:
        df = add_ma(df, window=w)

    df = add_macd(df)
    df = add_bollinger(df)
    df = add_rsi(df)

    return df
