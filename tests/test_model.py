"""Unit tests for recurrent neural network models and autoregressive projections."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.model import build_model, forecast_future


@pytest.mark.parametrize("arch", ["Stacked LSTM", "Bidirectional LSTM", "GRU"])
def test_build_model_architectures(arch: str) -> None:
    """Verify neural network compilation and input/output tensor contract."""
    lookback = 30
    model = build_model(architecture=arch, lookback=lookback)

    assert model is not None
    # Model should accept (batch_size, 30, 1) and output (batch_size, 1)
    dummy_input = np.random.uniform(0, 1, size=(4, lookback, 1)).astype(np.float32)
    output = model(dummy_input)
    assert output.shape == (4, 1)


def test_forecast_future_output_shape() -> None:
    """Verify autoregressive future projection returns correct dimensions."""
    lookback = 20
    model = build_model(architecture="Stacked LSTM", lookback=lookback)

    scaler = MinMaxScaler()
    scaler.fit(np.array([[100.0], [200.0]]))

    dummy_seq = np.linspace(0.1, 0.9, lookback)
    future_steps = 5
    forecast_df = forecast_future(
        model=model,
        last_sequence=dummy_seq,
        scaler=scaler,
        future_steps=future_steps,
        last_date=pd.Timestamp("2024-01-01"),
    )

    assert len(forecast_df) == future_steps
    assert "Projected Closing Price" in forecast_df.columns
    assert not forecast_df["Projected Closing Price"].isna().any()
