"""Data Preprocessing and Time-Series Sequence Engine for TrendPulse-AI.

Implements leakage-free chronological splitting, MinMax scaling, and
sliding-window sequence generation for recurrent neural network models.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.exceptions import InsufficientDataError, PreprocessingError
from src.logger import setup_logger

logger = setup_logger(__name__)


def validate_timeseries_data(df: pd.DataFrame, lookback: int, min_test_samples: int = 30) -> pd.DataFrame:
    """Validate that the dataset meets minimum volumetric and quality thresholds.

    Args:
        df: Input financial DataFrame.
        lookback: Number of past trading sessions used per prediction step.
        min_test_samples: Minimum number of samples required for the evaluation partition.

    Returns:
        Clean, validated DataFrame.

    Raises:
        InsufficientDataError: When records are below the required threshold.
        PreprocessingError: When required columns are missing or contain corrupt values.
    """
    if df is None or df.empty:
        raise InsufficientDataError("Downloaded market dataset is empty.")

    if "Close" not in df.columns:
        raise PreprocessingError("Market dataset missing primary target column 'Close'.")

    clean_df = df.copy().dropna(subset=["Close"])
    clean_df = clean_df[np.isfinite(clean_df["Close"])]

    required_minimum = lookback + min_test_samples
    if len(clean_df) < required_minimum:
        raise InsufficientDataError(
            f"Insufficient historical data: Found {len(clean_df)} trading days, "
            f"but at least {required_minimum} are required for lookback={lookback}."
        )

    return clean_df


def create_sequences(
    scaled_values: np.ndarray,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform sequential time-series into sliding lookback window tensors.

    Given a 1D or 2D array of length T:
    - Feature window X[i] contains samples from i-lookback to i-1.
    - Target y[i] contains the sample at index i.

    Args:
        scaled_values: Normalized array of shape (T,) or (T, 1).
        lookback: Sliding window length in trading sessions.

    Returns:
        Tuple of:
            X: 3D tensor of shape (N, lookback, 1)
            y: 1D array of shape (N,)
            target_indices: Array of original sample indices corresponding to y
    """
    values = np.asarray(scaled_values, dtype=np.float32).reshape(-1)
    total_length = len(values)

    if total_length <= lookback:
        raise PreprocessingError(
            f"Sequence length ({total_length}) must be strictly greater than lookback ({lookback})."
        )

    X_list, y_list, indices_list = [], [], []
    for idx in range(lookback, total_length):
        X_list.append(values[idx - lookback : idx])
        y_list.append(values[idx])
        indices_list.append(idx)

    X = np.asarray(X_list, dtype=np.float32).reshape(-1, lookback, 1)
    y = np.asarray(y_list, dtype=np.float32)
    target_indices = np.asarray(indices_list, dtype=int)

    logger.debug("Generated %d sequence samples with lookback=%d", len(X), lookback)
    return X, y, target_indices


def scale_split_data(
    close_prices: np.ndarray,
    train_split_ratio: float = 0.8,
) -> Tuple[MinMaxScaler, np.ndarray, int]:
    """Fit scaler strictly on training partition to eliminate lookahead data leakage.

    Args:
        close_prices: 1D array of closing prices.
        train_split_ratio: Percentage of data allocated to training (e.g. 0.8).

    Returns:
        Tuple of:
            scaler: Fitted MinMaxScaler instance
            scaled_series: Scaled 1D array across the entire timeline
            split_index: Boundary integer index separating train and test
    """
    total_samples = len(close_prices)
    split_index = int(total_samples * train_split_ratio)

    if split_index <= 0 or split_index >= total_samples:
        raise PreprocessingError(f"Invalid split index {split_index} for length {total_samples}")

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    train_partition = close_prices[:split_index].reshape(-1, 1)
    scaler.fit(train_partition)

    # Transform the entire timeline using only parameters learned from the training slice
    scaled_series = scaler.transform(close_prices.reshape(-1, 1)).flatten()

    logger.debug(
        "Fitted scaler on %d training samples. Min: %.2f, Max: %.2f",
        split_index,
        scaler.data_min_[0],
        scaler.data_max_[0],
    )
    return scaler, scaled_series, split_index
