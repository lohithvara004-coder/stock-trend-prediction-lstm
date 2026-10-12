"""Compatibility facade for src.data_handler."""

from src.data_handler import fetch_stock_data, get_ticker_metadata, validate_symbol

__all__ = ["fetch_stock_data", "get_ticker_metadata", "validate_symbol"]
