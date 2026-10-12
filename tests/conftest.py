"""Pytest fixtures and synthetic market time-series generators for TrendPulse-AI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_price_series() -> pd.Series:
    """Generate 200 trading days of simulated price data via geometric walk."""
    np.random.seed(42)
    dates = pd.bdate_range(start="2023-01-01", periods=200)
    # Generate drift + random walk
    returns = np.random.normal(loc=0.0005, scale=0.015, size=200)
    prices = 150.0 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates, name="Close")


@pytest.fixture
def sample_market_df(sample_price_series: pd.Series) -> pd.DataFrame:
    """Generate full OHLCV dataframe matching yfinance schema."""
    close = sample_price_series.values
    high = close * (1.0 + np.random.uniform(0.002, 0.02, size=len(close)))
    low = close * (1.0 - np.random.uniform(0.002, 0.02, size=len(close)))
    open_p = low + (high - low) * np.random.uniform(0.2, 0.8, size=len(close))
    volume = np.random.randint(1_000_000, 10_000_000, size=len(close))

    df = pd.DataFrame(
        {
            "Open": open_p,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=sample_price_series.index,
    )
    df.index.name = "Date"
    return df
