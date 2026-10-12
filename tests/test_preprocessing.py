"""Unit tests for time-series preprocessing, sequence generation, and zero-leakage scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.exceptions import InsufficientDataError, PreprocessingError
from src.preprocessing import create_sequences, scale_split_data, validate_timeseries_data


def test_validate_timeseries_data_success(sample_market_df: pd.DataFrame) -> None:
    """Validate healthy dataset passing validation checks."""
    result = validate_timeseries_data(sample_market_df, lookback=60, min_test_samples=30)
    assert len(result) == len(sample_market_df)


def test_validate_timeseries_insufficient_samples() -> None:
    """Ensure InsufficientDataError is raised on too few records."""
    tiny_df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})
    with pytest.raises(InsufficientDataError):
        validate_timeseries_data(tiny_df, lookback=60)


def test_create_sequences_shape_and_values() -> None:
    """Verify 3D tensor shape and chronological windowing correctness."""
    synthetic_prices = np.arange(100, dtype=np.float32)
    lookback = 10
    X, y, indices = create_sequences(synthetic_prices, lookback=lookback)

    # N = 100 - 10 = 90
    assert X.shape == (90, 10, 1)
    assert y.shape == (90,)
    assert indices.shape == (90,)

    # First sequence window: 0 through 9, target is 10
    assert np.array_equal(X[0].flatten(), np.arange(10, dtype=np.float32))
    assert y[0] == 10.0
    assert indices[0] == 10


def test_scale_split_zero_data_leakage(sample_price_series: pd.Series) -> None:
    """Confirm scaler parameters are derived purely from training slice."""
    prices = sample_price_series.values
    train_ratio = 0.8
    split_index = int(len(prices) * train_ratio)

    scaler, scaled_series, split_idx = scale_split_data(prices, train_split_ratio=train_ratio)

    assert split_idx == split_index
    assert len(scaled_series) == len(prices)

    # Scaler min and max must match the min and max of ONLY the training slice
    train_slice = prices[:split_index]
    assert np.isclose(scaler.data_min_[0], train_slice.min())
    assert np.isclose(scaler.data_max_[0], train_slice.max())
