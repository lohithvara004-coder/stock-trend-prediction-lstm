"""Quantitative Performance Metrics for TrendPulse-AI.

Calculates statistical error and directional predictability metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Directional Accuracy (Hit Ratio / Trend Direction %)
- Coefficient of Determination (R-squared)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, float]:
    """Compute comprehensive time-series evaluation metrics.

    Args:
        actual: 1D ground-truth price array.
        predicted: 1D model predicted price array.

    Returns:
        Dictionary mapping metric names to computed floating point values.
    """
    actual_arr = np.asarray(actual, dtype=np.float64).flatten()
    pred_arr = np.asarray(predicted, dtype=np.float64).flatten()

    if len(actual_arr) != len(pred_arr):
        raise ValueError(f"Array length mismatch: actual={len(actual_arr)}, pred={len(pred_arr)}")
    if len(actual_arr) < 2:
        raise ValueError("At least 2 points are required to calculate metrics.")

    # Statistical Errors
    mae = float(mean_absolute_error(actual_arr, pred_arr))
    mse = float(mean_squared_error(actual_arr, pred_arr))
    rmse = float(np.sqrt(mse))

    # MAPE with epsilon to prevent zero division
    epsilon = 1e-8
    mape = float(np.mean(np.abs((actual_arr - pred_arr) / (actual_arr + epsilon))) * 100.0)

    # R2 Score
    r2 = float(r2_score(actual_arr, pred_arr))

    # Directional Accuracy (Hit Ratio):
    # Tests whether the model predicted the correct upward or downward movement relative to yesterday
    actual_direction = np.sign(actual_arr[1:] - actual_arr[:-1])
    predicted_direction = np.sign(pred_arr[1:] - actual_arr[:-1])
    correct_directions = np.sum(actual_direction == predicted_direction)
    directional_accuracy = float((correct_directions / len(actual_direction)) * 100.0)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2),
        "r2": round(r2, 4),
        "directional_accuracy": round(directional_accuracy, 2),
    }
