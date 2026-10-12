"""Deep Learning Time-Series Forecasting Engine for TrendPulse-AI.

Implements modular recurrent neural network architectures (Stacked LSTM,
Bidirectional LSTM, and GRU) with early stopping, learning rate scheduling,
and multi-step forward autoregressive projection.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, GRU, Input, LSTM
from tensorflow.keras.optimizers import Adam

from src.config import config
from src.exceptions import ModelTrainingError, PredictionError
from src.logger import setup_logger
from src.metrics import calculate_metrics
from src.preprocessing import create_sequences, scale_split_data

logger = setup_logger(__name__)


def build_model(
    architecture: str = "Stacked LSTM",
    lookback: int = 60,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.2,
) -> Sequential:
    """Instantiate and compile deep recurrent neural network architecture.

    Args:
        architecture: One of 'Stacked LSTM', 'Bidirectional LSTM', or 'GRU'.
        lookback: Temporal lookback window length.
        learning_rate: Optimizer Adam learning rate.
        dropout_rate: Dropout probability to combat overfitting.

    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential()
    model.add(Input(shape=(lookback, 1)))

    if architecture == "Bidirectional LSTM":
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(dropout_rate))
    elif architecture == "GRU":
        model.add(GRU(64, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(GRU(32, return_sequences=False))
        model.add(Dropout(dropout_rate))
    else:  # Default to Stacked LSTM
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(dropout_rate))

    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])

    logger.debug("Successfully built %s architecture (lookback=%d)", architecture, lookback)
    return model


def forecast_future(
    model: Sequential,
    last_sequence: np.ndarray,
    scaler: Any,
    future_steps: int = 7,
    last_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Generate forward-looking autoregressive projections for future trading days.

    Args:
        model: Trained recurrent neural network model.
        last_sequence: Normalized 1D array of length `lookback`.
        scaler: Fitted MinMaxScaler used during training.
        future_steps: Number of forward trading sessions to project.
        last_date: Last known trading date in the historical dataset.

    Returns:
        DataFrame containing forecasted dates and projected closing prices.
    """
    current_seq = np.asarray(last_sequence, dtype=np.float32).reshape(1, -1, 1)
    predictions = []

    for _ in range(future_steps):
        next_pred_scaled = model.predict(current_seq, verbose=0)
        val = float(np.squeeze(next_pred_scaled))
        predictions.append(val)

        # Shift sequence window left and append new prediction
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, 0] = val

    preds_scaled = np.array(predictions).reshape(-1, 1)
    preds_dollar = scaler.inverse_transform(preds_scaled).flatten()

    # Generate future business days
    start_dt = pd.Timestamp.now() if last_date is None else pd.Timestamp(last_date)
    future_dates = pd.bdate_range(start=start_dt + pd.Timedelta(days=1), periods=future_steps)

    forecast_df = pd.DataFrame(
        {"Projected Closing Price": preds_dollar},
        index=future_dates,
    )
    forecast_df.index.name = "Date"
    return forecast_df


def train_and_predict(
    data: pd.DataFrame,
    lookback: int = 60,
    epochs: int = 15,
    architecture: str = "Stacked LSTM",
    train_split_ratio: float = 0.8,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Dict[str, Any]:
    """Execute end-to-end model training, validation, testing, and forward forecasting.

    Args:
        data: Historical DataFrame with 'Close' column.
        lookback: Input time window size.
        epochs: Training iterations.
        architecture: Selected neural network architecture.
        train_split_ratio: Ratio for chronological train/test partitioning.
        progress_callback: Optional callback receiving (current_epoch, total_epochs, loss).

    Returns:
        Dictionary containing evaluation table, metrics, history, future forecast, and model.
    """
    close_prices = data["Close"].to_numpy(dtype=np.float32)

    try:
        scaler, scaled_series, split_index = scale_split_data(close_prices, train_split_ratio)
        X, y, target_indices = create_sequences(scaled_series, lookback)
    except Exception as exc:
        raise ModelTrainingError("Failed during data scaling or sequence creation", str(exc)) from exc

    train_mask = target_indices < split_index
    test_mask = target_indices >= split_index

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ModelTrainingError(
            "Insufficient sequences to populate both training and test partitions. "
            f"Train count: {len(X_train)}, Test count: {len(X_test)}"
        )

    logger.info(
        "Initiating %s training: %d train samples, %d test samples, %d epochs",
        architecture,
        len(X_train),
        len(X_test),
        epochs,
    )

    model = build_model(
        architecture=architecture,
        lookback=lookback,
        learning_rate=config.model.learning_rate,
        dropout_rate=config.model.dropout_rate,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config.model.patience,
            restore_best_weights=True,
            mode="min",
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=0,
        ),
    ]

    if progress_callback:
        class EpochReportingCallback(Callback):
            def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
                current_loss = logs.get("loss", 0.0) if logs else 0.0
                progress_callback(epoch + 1, epochs, float(current_loss))

        callbacks.append(EpochReportingCallback())

    val_split = config.model.validation_split_ratio if len(X_train) >= 30 else 0.0

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=config.model.batch_size,
        shuffle=False,  # Preserve temporal sequence order
        validation_split=val_split,
        callbacks=callbacks,
        verbose=0,
    )

    # Test Partition Inference
    try:
        pred_scaled = model.predict(X_test, verbose=0)
        pred_prices = scaler.inverse_transform(pred_scaled).flatten()
        actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        test_dates = data.index[target_indices[test_mask]]
    except Exception as exc:
        raise PredictionError("Inference failed on test partition", str(exc)) from exc

    table = pd.DataFrame(
        {
            "Actual Closing Price": np.round(actual_prices, 2),
            "Predicted Closing Price": np.round(pred_prices, 2),
            "Absolute Error ($)": np.round(np.abs(actual_prices - pred_prices), 2),
            "Percentage Error (%)": np.round(
                (np.abs(actual_prices - pred_prices) / (actual_prices + 1e-8)) * 100.0, 2
            ),
        },
        index=test_dates,
    )
    table.index.name = "Date"

    # Compute Statistical Metrics
    metrics_summary = calculate_metrics(actual_prices, pred_prices)

    # 7-Day Autoregressive Future Projection
    last_lookback_sequence = scaled_series[-lookback:]
    future_forecast = forecast_future(
        model=model,
        last_sequence=last_lookback_sequence,
        scaler=scaler,
        future_steps=7,
        last_date=data.index[-1],
    )

    return {
        "table": table,
        "metrics": metrics_summary,
        "mae": metrics_summary["mae"],
        "rmse": metrics_summary["rmse"],
        "mape": metrics_summary["mape"],
        "directional_accuracy": metrics_summary["directional_accuracy"],
        "r2": metrics_summary["r2"],
        "history": history.history,
        "future_forecast": future_forecast,
        "model": model,
        "scaler": scaler,
    }
