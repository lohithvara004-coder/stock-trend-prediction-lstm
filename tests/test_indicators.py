"""Unit tests for quantitative technical indicator calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators import (
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
    compute_all_indicators,
)


def test_calculate_sma(sample_price_series: pd.Series) -> None:
    """Verify Simple Moving Average calculation and length preservation."""
    sma_20 = calculate_sma(sample_price_series, window=20)
    assert len(sma_20) == len(sample_price_series)
    # The 20th entry should match arithmetic mean of first 20 days
    expected = sample_price_series.iloc[:20].mean()
    assert np.isclose(sma_20.iloc[19], expected, rtol=1e-5)


def test_calculate_ema(sample_price_series: pd.Series) -> None:
    """Verify Exponential Moving Average properties."""
    ema_12 = calculate_ema(sample_price_series, window=12)
    assert len(ema_12) == len(sample_price_series)
    assert not ema_12.isna().any()


def test_calculate_rsi_bounds(sample_price_series: pd.Series) -> None:
    """Verify RSI remains bounded between 0 and 100."""
    rsi = calculate_rsi(sample_price_series, window=14)
    assert len(rsi) == len(sample_price_series)
    assert (rsi >= 0.0).all()
    assert (rsi <= 100.0).all()


def test_calculate_bollinger_bands_geometry(sample_price_series: pd.Series) -> None:
    """Verify upper band >= middle band >= lower band invariant."""
    upper, mid, lower = calculate_bollinger_bands(sample_price_series, window=20, num_std=2.0)
    assert len(upper) == len(sample_price_series)
    assert (upper >= mid).all()
    assert (mid >= lower).all()


def test_calculate_macd_structure(sample_price_series: pd.Series) -> None:
    """Verify MACD line, signal line, and histogram arithmetic consistency."""
    macd, signal, hist = calculate_macd(sample_price_series, fast=12, slow=26, signal=9)
    assert len(macd) == len(sample_price_series)
    assert len(signal) == len(sample_price_series)
    assert len(hist) == len(sample_price_series)
    # Histogram must equal MACD - Signal
    assert np.allclose(hist.values, (macd - signal).values, atol=1e-7)


def test_compute_all_indicators(sample_market_df: pd.DataFrame) -> None:
    """Verify all expected columns are injected into the dataframe."""
    enriched = compute_all_indicators(sample_market_df)
    expected_cols = [
        "SMA_20", "SMA_50", "SMA_100", "SMA_200",
        "EMA_12", "EMA_26", "RSI_14",
        "BB_Upper", "BB_Middle", "BB_Lower",
        "MACD", "MACD_Signal", "MACD_Hist"
    ]
    for col in expected_cols:
        assert col in enriched.columns
