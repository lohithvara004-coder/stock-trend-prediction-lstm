"""Unit tests for quantitative evaluation metrics calculation."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import calculate_metrics


def test_calculate_metrics_perfect_predictions() -> None:
    """Verify metrics when predictions perfectly match ground truth."""
    actual = np.array([100.0, 105.0, 110.0, 108.0, 115.0])
    predicted = np.array([100.0, 105.0, 110.0, 108.0, 115.0])

    metrics = calculate_metrics(actual, predicted)

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mape"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["directional_accuracy"] == 100.0


def test_calculate_metrics_known_error() -> None:
    """Verify metrics with constant absolute error offset."""
    actual = np.array([100.0, 110.0, 120.0, 130.0])
    predicted = np.array([102.0, 112.0, 122.0, 132.0])

    metrics = calculate_metrics(actual, predicted)

    assert np.isclose(metrics["mae"], 2.0)
    assert np.isclose(metrics["rmse"], 2.0)
    assert metrics["directional_accuracy"] == 100.0


def test_calculate_metrics_mismatched_lengths() -> None:
    """Ensure ValueError is raised on mismatched input lengths."""
    actual = np.array([100.0, 105.0])
    predicted = np.array([100.0])

    with pytest.raises(ValueError):
        calculate_metrics(actual, predicted)
