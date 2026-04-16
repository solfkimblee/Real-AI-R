"""Streamlit-cached wrappers around expensive data calls.

Streamlit reruns the whole script on every widget change; without caching,
every A-share fetch and model train goes back to AKShare / disk. Wrap the
hot paths here and import from pages_*.py instead of hitting DataFetcher
directly.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from real_ai_r.data.fetcher import DataFetcher


@st.cache_resource(show_spinner=False)
def get_fetcher() -> DataFetcher:
    """One DataFetcher per Streamlit server process."""
    return DataFetcher()


@st.cache_data(ttl=900, show_spinner="加载日线数据…")
def load_stock_daily(
    symbol: str,
    start: str,
    end: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """A-share EOD data — safe to cache 15 min (market data is EOD)."""
    return get_fetcher().get_stock_daily(
        symbol=symbol, start_date=start, end_date=end, adjust=adjust
    )
