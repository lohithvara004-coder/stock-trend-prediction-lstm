"""Compatibility facade for src.model."""

from src.model import build_model, forecast_future, train_and_predict

# Backwards compatibility alias
build_lstm_model = lambda lookback: build_model("Stacked LSTM", lookback)

__all__ = ["build_model", "build_lstm_model", "forecast_future", "train_and_predict"]
