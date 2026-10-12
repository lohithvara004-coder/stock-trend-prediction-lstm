"""Quantitative Technical Indicators Engine for TrendPulse-AI.

Implements mathematical and statistical time-series indicators:
- Simple Moving Averages (SMA: 20, 50, 100, 200)
- Exponential Moving Averages (EMA: 12, 26)
- Relative Strength Index (RSI: 14)
- Bollinger Bands (Upper, Middle, Lower: 20, 2.0 std)
- Moving Average Convergence Divergence (MACD, Signal, Histogram)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.logger import setup_logger

logger = setup_logger(__name__)


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Compute Simple Moving Average over a specified rolling window.

    Args:
        series: Pandas series of numerical values (e.g., closing prices).
        window: Rolling lookback period in days.

    Returns:
        pd.Series containing the calculated SMA.
    """
    if window <= 0:
        raise ValueError("Window size must be a positive integer.")
    return series.rolling(window=window, min_periods=1).mean()


def calculate_ema(series: pd.Series, window: int) -> pd.Series:
    """Compute Exponential Moving Average over a specified rolling window.

    Args:
        series: Pandas series of numerical values.
        window: Window span for exponential weighting.

    Returns:
        pd.Series containing the calculated EMA.
    """
    if window <= 0:
        raise ValueError("Window size must be a positive integer.")
    return series.ewm(span=window, adjust=False).mean()


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI) using Wilder's smoothed method.

    Args:
        series: Closing price series.
        window: Lookback period (standard is 14 days).

    Returns:
        pd.Series with RSI values bounded between [0, 100].
    """
    if window <= 0:
        raise ValueError("Window size must be a positive integer.")

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's exponential smoothing
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    # Prevent division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Fill initial boundary conditions
    rsi = rsi.fillna(50.0)
    return rsi.clip(lower=0.0, upper=100.0)


def calculate_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands (Upper, Middle, Lower).

    Args:
        series: Closing price series.
        window: Moving average rolling period.
        num_std: Standard deviation multiplier.

    Returns:
        Tuple of (upper_band, middle_band, lower_band).
    """
    if window <= 0:
        raise ValueError("Window size must be a positive integer.")
    if num_std <= 0:
        raise ValueError("Standard deviation multiplier must be positive.")

    middle = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std(ddof=0).fillna(0.0)
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower


def calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Moving Average Convergence Divergence (MACD).

    Args:
        series: Closing price series.
        fast: Short-period EMA window.
        slow: Long-period EMA window.
        signal: Signal line EMA window.

    Returns:
        Tuple of (macd_line, signal_line, histogram).
    """
    if fast >= slow:
        raise ValueError("Fast window must be strictly less than slow window.")

    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich the input financial DataFrame with all technical indicators.

    Args:
        df: DataFrame containing at minimum a 'Close' column.

    Returns:
        A new DataFrame enriched with technical indicator feature columns.
    """
    if "Close" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Close' column.")

    result = df.copy()
    close = result["Close"]

    logger.debug("Computing quantitative indicators for %d trading sessions", len(df))

    # Simple Moving Averages
    result["SMA_20"] = calculate_sma(close, 20)
    result["SMA_50"] = calculate_sma(close, 50)
    result["SMA_100"] = calculate_sma(close, 100)
    result["SMA_200"] = calculate_sma(close, 200)

    # Exponential Moving Averages
    result["EMA_12"] = calculate_ema(close, 12)
    result["EMA_26"] = calculate_ema(close, 26)

    # RSI
    result["RSI_14"] = calculate_rsi(close, 14)

    # Bollinger Bands
    upper, mid, lower = calculate_bollinger_bands(close, window=20, num_std=2.0)
    result["BB_Upper"] = upper
    result["BB_Middle"] = mid
    result["BB_Lower"] = lower

    # MACD
    macd, sig, hist = calculate_macd(close, fast=12, slow=26, signal=9)
    result["MACD"] = macd
    result["MACD_Signal"] = sig
    result["MACD_Hist"] = hist

    return result
