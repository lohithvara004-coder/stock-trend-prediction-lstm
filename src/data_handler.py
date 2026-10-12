"""Market Data Ingestion and Validation Engine for TrendPulse-AI.

Interacts with financial market APIs (Yahoo Finance), sanitizes historical
quotes, manages caching, and provides structured market metadata.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

import pandas as pd
import yfinance as yf

from src.config import config
from src.exceptions import DataFetchError, InsufficientDataError
from src.logger import setup_logger

logger = setup_logger(__name__)


def validate_symbol(symbol: str) -> str:
    """Sanitize and format market ticker symbol.

    Args:
        symbol: Raw user-provided ticker string.

    Returns:
        Clean uppercase ticker string.

    Raises:
        ValueError: If symbol is empty or invalid.
    """
    cleaned = symbol.strip().upper()
    if not cleaned:
        raise ValueError("Stock symbol cannot be empty.")
    return cleaned


def fetch_stock_data(
    ticker: str,
    start_date: str | date,
    end_date: str | date,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Fetch and sanitize historical daily market data.

    Yahoo Finance's `end` parameter is non-inclusive, so one day is appended
    to capture the complete calendar bounds.

    Args:
        ticker: Ticker symbol (e.g. AAPL, NVDA, MSFT, RELIANCE.NS).
        start_date: Start date string (YYYY-MM-DD) or date object.
        end_date: End date string (YYYY-MM-DD) or date object.
        auto_adjust: Whether to adjust Close prices for stock splits and dividends.

    Returns:
        pd.DataFrame indexed by Date containing Open, High, Low, Close, Volume.

    Raises:
        DataFetchError: If data retrieval encounters an unrecoverable failure.
    """
    clean_ticker = validate_symbol(ticker)
    start_str = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    # Add one day so the end date is inclusive
    end_exclusive = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(
        "Downloading financial time-series for %s [%s -> %s]",
        clean_ticker,
        start_str,
        end_str := pd.Timestamp(end_date).strftime("%Y-%m-%d"),
    )

    try:
        raw_df = yf.download(
            tickers=clean_ticker,
            start=start_str,
            end=end_exclusive,
            auto_adjust=auto_adjust,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        logger.error("Failed to fetch market data for %s: %s", clean_ticker, exc)
        raise DataFetchError(f"Network or provider error fetching ticker '{clean_ticker}'", str(exc)) from exc

    if raw_df is None or raw_df.empty:
        logger.warning("No records returned for symbol: %s", clean_ticker)
        return pd.DataFrame()

    # Flatten potential MultiIndex column headers produced by recent yfinance releases
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)

    # De-duplicate any redundant column headers
    df = raw_df.loc[:, ~raw_df.columns.duplicated()].copy()

    if "Close" not in df.columns:
        logger.error("Required column 'Close' missing in downloaded payload for %s", clean_ticker)
        return pd.DataFrame()

    # Coerce numerical columns and strip NaNs
    numeric_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"]).copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df.sort_index(inplace=True)

    logger.info("Successfully ingested %d trading sessions for %s", len(df), clean_ticker)
    return df


def get_ticker_metadata(ticker: str) -> Dict[str, Any]:
    """Retrieve asset metadata (Company name, sector, currency, summary).

    Args:
        ticker: Ticker symbol.

    Returns:
        Dictionary of company metadata.
    """
    clean_ticker = validate_symbol(ticker)
    try:
        t = yf.Ticker(clean_ticker)
        info = t.info or {}
        return {
            "symbol": clean_ticker,
            "shortName": info.get("shortName") or info.get("longName") or clean_ticker,
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
            "marketCap": info.get("marketCap", 0),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", 0.0),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", 0.0),
            "regularMarketPrice": info.get("regularMarketPrice", 0.0),
        }
    except Exception as exc:
        logger.debug("Could not retrieve extended metadata for %s: %s", clean_ticker, exc)
        return {
            "symbol": clean_ticker,
            "shortName": clean_ticker,
            "sector": "N/A",
            "industry": "N/A",
            "currency": "USD",
            "exchange": "N/A",
            "marketCap": 0,
            "fiftyTwoWeekHigh": 0.0,
            "fiftyTwoWeekLow": 0.0,
            "regularMarketPrice": 0.0,
        }
