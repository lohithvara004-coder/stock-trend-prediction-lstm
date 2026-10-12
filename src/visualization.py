"""Interactive Financial Data Visualization Engine for TrendPulse-AI.

Generates high-performance interactive Plotly charts:
- Candlestick charts with volume subplots and moving averages
- Multi-panel technical indicator dashboards (RSI, MACD)
- Actual vs Predicted price comparisons with error bands
- Training loss convergence curves
- Forward-looking trend projections
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Construct an interactive Candlestick chart with volume and moving average overlays.

    Args:
        df: DataFrame containing Open, High, Low, Close, Volume, and SMA indicators.
        ticker: Asset symbol string.

    Returns:
        Plotly Figure object.
    """
    has_ohlc = all(c in df.columns for c in ["Open", "High", "Low", "Close"])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{ticker} Price Action & Trend Averages", "Trading Volume"),
    )

    if has_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                name="Close Price",
                line=dict(color="#2962ff", width=2),
            ),
            row=1,
            col=1,
        )

    # Moving Average Overlays
    if "SMA_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_50"],
                name="SMA 50",
                line=dict(color="#ff9800", width=1.5),
            ),
            row=1,
            col=1,
        )
    if "SMA_200" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_200"],
                name="SMA 200",
                line=dict(color="#e91e63", width=1.5),
            ),
            row=1,
            col=1,
        )

    # Bollinger Bands
    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Upper"],
                name="BB Upper",
                line=dict(color="rgba(150, 150, 150, 0.4)", width=1, dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Lower"],
                name="BB Lower",
                line=dict(color="rgba(150, 150, 150, 0.4)", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(150, 150, 150, 0.08)",
            ),
            row=1,
            col=1,
        )

    # Volume subplot
    if "Volume" in df.columns:
        colors = [
            "#26a69a" if (df["Close"].iloc[i] >= df["Open"].iloc[i] if has_ohlc else True) else "#ef5350"
            for i in range(len(df))
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.6,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=580,
    )
    return fig


def create_technical_indicators_chart(df: pd.DataFrame) -> go.Figure:
    """Create a 2-panel dashboard for Relative Strength Index and MACD.

    Args:
        df: DataFrame containing computed RSI_14, MACD, MACD_Signal, MACD_Hist.

    Returns:
        Plotly Figure object.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Relative Strength Index (RSI 14)", "MACD (12, 26, 9)"),
    )

    # Row 1: RSI
    if "RSI_14" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI_14"],
                name="RSI 14",
                line=dict(color="#ab47bc", width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_hline(
            y=70, line_dash="dash", line_color="#ef5350", annotation_text="Overbought (70)", row=1, col=1
        )
        fig.add_hline(
            y=30, line_dash="dash", line_color="#26a69a", annotation_text="Oversold (30)", row=1, col=1
        )

    # Row 2: MACD
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD"],
                name="MACD Line",
                line=dict(color="#29b6f6", width=1.8),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_Signal"],
                name="Signal Line",
                line=dict(color="#ffa726", width=1.8),
            ),
            row=2,
            col=1,
        )
        if "MACD_Hist" in df.columns:
            hist_colors = ["#26a69a" if val >= 0 else "#ef5350" for val in df["MACD_Hist"]]
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df["MACD_Hist"],
                    name="MACD Histogram",
                    marker_color=hist_colors,
                ),
                row=2,
                col=1,
            )

    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
    )
    return fig


def create_prediction_comparison_chart(table: pd.DataFrame, ticker: str) -> go.Figure:
    """Generate high-precision actual versus predicted comparison chart with error ribbon.

    Args:
        table: DataFrame containing 'Actual Closing Price' and 'Predicted Closing Price'.
        ticker: Asset symbol string.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    actual = table["Actual Closing Price"]
    predicted = table["Predicted Closing Price"]

    # Ground truth
    fig.add_trace(
        go.Scatter(
            x=table.index,
            y=actual,
            name="Actual Market Price",
            line=dict(color="#00e676", width=2.2),
            mode="lines",
        )
    )

    # Neural network forecast
    fig.add_trace(
        go.Scatter(
            x=table.index,
            y=predicted,
            name="Deep Learning Prediction",
            line=dict(color="#ff3d00", width=2.0, dash="solid"),
            mode="lines",
        )
    )

    fig.update_layout(
        title=f"{ticker} — Model Evaluation on Held-Out Test Period",
        xaxis_title="Trading Date",
        yaxis_title="Stock Price ($)",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=520,
    )
    return fig


def create_loss_curve_chart(history: Dict[str, Any]) -> go.Figure:
    """Plot neural network training and validation loss convergence.

    Args:
        history: Training history dictionary with 'loss' and optional 'val_loss'.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    epochs = list(range(1, len(history.get("loss", [])) + 1))

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history.get("loss", []),
            name="Training Loss (MSE)",
            line=dict(color="#29b6f6", width=2),
            mode="lines+markers",
        )
    )

    if "val_loss" in history and history["val_loss"]:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history.get("val_loss", []),
                name="Validation Loss (MSE)",
                line=dict(color="#ffca28", width=2),
                mode="lines+markers",
            )
        )

    fig.update_layout(
        title="Neural Network Loss Convergence (Optimization Trajectory)",
        xaxis_title="Epoch",
        yaxis_title="Mean Squared Error Loss",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
    )
    return fig


def create_future_forecast_chart(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    ticker: str,
    lookback_display: int = 60,
) -> go.Figure:
    """Plot recent historical prices connected seamlessly to future projected sessions.

    Args:
        historical_df: DataFrame of historical prices.
        forecast_df: DataFrame of autoregressively projected future prices.
        ticker: Symbol string.
        lookback_display: Number of recent historical days to display for context.

    Returns:
        Plotly Figure object.
    """
    recent_hist = historical_df.tail(lookback_display)
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recent_hist.index,
            y=recent_hist["Close"],
            name="Recent Historical Close",
            line=dict(color="#90caf9", width=2),
            mode="lines",
        )
    )

    # Stitch the last historical point with the projection for continuity
    stitch_dates = [recent_hist.index[-1]] + list(forecast_df.index)
    stitch_prices = [recent_hist["Close"].iloc[-1]] + list(forecast_df["Projected Closing Price"])

    fig.add_trace(
        go.Scatter(
            x=stitch_dates,
            y=stitch_prices,
            name="Projected Trend (Next 7 Sessions)",
            line=dict(color="#ffd600", width=2.5, dash="dash"),
            marker=dict(size=6, color="#ffd600"),
            mode="lines+markers",
        )
    )

    fig.update_layout(
        title=f"{ticker} — 7-Day Forward Autoregressive Trend Projection",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450,
    )
    return fig
