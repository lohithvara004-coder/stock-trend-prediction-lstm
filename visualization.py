"""Compatibility facade for src.visualization."""

import matplotlib.pyplot as plt
import streamlit as st

from src.visualization import (
    create_candlestick_chart,
    create_future_forecast_chart,
    create_loss_curve_chart,
    create_prediction_comparison_chart,
    create_technical_indicators_chart,
)


def show_historical_chart(data):
    """Legacy Matplotlib historical visualization."""
    chart_data = data.copy()
    chart_data["MA 100"] = chart_data["Close"].rolling(window=100).mean()
    chart_data["MA 200"] = chart_data["Close"].rolling(window=200).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(chart_data.index, chart_data["Close"], label="Closing price")
    ax.plot(chart_data.index, chart_data["MA 100"], label="100-day moving average")
    ax.plot(chart_data.index, chart_data["MA 200"], label="200-day moving average")
    ax.set_title("Historical Closing Price and Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_prediction_chart(results):
    """Legacy Matplotlib test prediction visualization."""
    table = results["table"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(table.index, table["Actual Closing Price"], label="Actual price")
    ax.plot(table.index, table["Predicted Closing Price"], label="Predicted price")
    ax.set_title("Actual vs Predicted Closing Prices (Historical Test Period)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


__all__ = [
    "create_candlestick_chart",
    "create_future_forecast_chart",
    "create_loss_curve_chart",
    "create_prediction_comparison_chart",
    "create_technical_indicators_chart",
    "show_historical_chart",
    "show_prediction_chart",
]
