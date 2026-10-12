"""Configuration module for TrendPulse-AI.

Centralizes runtime parameters, hyperparameters, and application defaults
with environment variable overrides and strict type hinting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters and architectural configurations for time-series forecasting."""

    default_lookback: int = 60
    min_lookback: int = 15
    max_lookback: int = 180
    default_epochs: int = 15
    min_epochs: int = 5
    max_epochs: int = 100
    batch_size: int = 32
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    train_split_ratio: float = 0.8
    validation_split_ratio: float = 0.1
    patience: int = 4
    supported_models: List[str] = field(
        default_factory=lambda: ["Stacked LSTM", "Bidirectional LSTM", "GRU"]
    )


@dataclass(frozen=True)
class TechnicalIndicatorConfig:
    """Configuration for financial quantitative indicators."""

    sma_fast_window: int = 20
    sma_medium_window: int = 50
    sma_slow_window: int = 100
    sma_long_window: int = 200
    rsi_window: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    bollinger_window: int = 20
    bollinger_std_dev: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9


@dataclass(frozen=True)
class AppConfig:
    """Primary application configuration."""

    app_name: str = "TrendPulse-AI"
    app_version: str = "1.0.0"
    app_tagline: str = (
        "Enterprise Time-Series Stock Trend Prediction & Quantitative Analytics Engine"
    )
    author: str = "Lohith Varanasi"
    github_url: str = "https://github.com/lohithvara004-coder"
    default_ticker: str = os.getenv("DEFAULT_TICKER", "AAPL")
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    environment: str = os.getenv("APP_ENV", "production")

    model: ModelConfig = field(default_factory=ModelConfig)
    indicators: TechnicalIndicatorConfig = field(
        default_factory=TechnicalIndicatorConfig
    )

    @property
    def default_start_date(self) -> date:
        """Default start date: 5 years prior to today."""
        return date.today() - timedelta(days=5 * 365)

    @property
    def default_end_date(self) -> date:
        """Default end date: current calendar day."""
        return date.today()


config = AppConfig()
