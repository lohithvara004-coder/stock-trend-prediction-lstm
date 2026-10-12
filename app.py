"""TrendPulse-AI: Enterprise Stock Trend Prediction & Quantitative Analytics Engine.

Main Web Application Entrypoint powered by Streamlit and Plotly.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import streamlit as st
import pandas as pd

from src.config import config
from src.data_handler import fetch_stock_data, get_ticker_metadata
from src.exceptions import InsufficientDataError, PreprocessingError, TrendPulseException
from src.indicators import compute_all_indicators
from src.logger import setup_logger
from src.model import train_and_predict
from src.preprocessing import validate_timeseries_data
from src.visualization import (
    create_candlestick_chart,
    create_future_forecast_chart,
    create_loss_curve_chart,
    create_prediction_comparison_chart,
    create_technical_indicators_chart,
)

logger = setup_logger(__name__)

# Page Configuration
st.set_page_config(
    page_title=f"{config.app_name} | Enterprise Stock Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Enterprise CSS Styling
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e2638 0%, #151a28 100%);
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        color: #a0aec0;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #2b6cb0;
        color: #ebf8ff;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
col_head_1, col_head_2 = st.columns([0.75, 0.25])
with col_head_1:
    st.title("📈 TrendPulse-AI")
    st.caption(
        f"**{config.app_tagline}** | Architected by [{config.author}]({config.github_url})"
    )
with col_head_2:
    st.markdown(
        """
        <div style="text-align: right; margin-top: 15px;">
            <span class="badge-pill">Production Ready</span>
            <span class="badge-pill">TensorFlow 2.20</span>
            <span class="badge-pill">Plotly Dynamic</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Market & Model Settings")

    ticker_input = st.text_input(
        "Asset Ticker Symbol",
        value=config.default_ticker,
        help="Enter global equity ticker. Examples: AAPL, NVDA, MSFT, GOOGL, TSLA, RELIANCE.NS, TCS.NS",
    ).strip().upper()

    col_date_1, col_date_2 = st.columns(2)
    with col_date_1:
        start_date = st.date_input(
            "Start Date",
            value=config.default_start_date,
            max_value=date.today(),
        )
    with col_date_2:
        end_date = st.date_input(
            "End Date",
            value=config.default_end_date,
            min_value=start_date,
            max_value=date.today(),
        )

    st.subheader("🧠 Neural Network Parameters")
    model_architecture = st.selectbox(
        "Architecture",
        options=config.model.supported_models,
        index=0,
        help="Select deep recurrent neural network configuration.",
    )

    lookback = st.slider(
        "Lookback Window (Days)",
        min_value=config.model.min_lookback,
        max_value=config.model.max_lookback,
        value=config.model.default_lookback,
        step=5,
        help="Number of historical trading sessions used to predict the next price step.",
    )

    epochs = st.slider(
        "Training Epochs",
        min_value=config.model.min_epochs,
        max_value=config.model.max_epochs,
        value=config.model.default_epochs,
        step=5,
        help="Maximum training iterations (equipped with EarlyStopping).",
    )

    execute_btn = st.button("🚀 Ingest Data & Train Model", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown(
        """
        **System Features**
        - Zero Lookahead Leakage Scaling
        - Dynamic Technical Indicators (RSI, MACD, Bollinger)
        - Multi-step Autoregressive Projection
        - Enterprise Test Coverage & CI/CD
        """
    )

# Primary Application Execution Workflow
if execute_btn:
    if not ticker_input:
        st.error("Please enter a valid stock ticker symbol.")
        st.stop()
    if start_date >= end_date:
        st.error("End date must occur strictly after start date.")
        st.stop()

    # Step 1: Market Data Ingestion
    with st.spinner(f"Ingesting market quotes for {ticker_input}..."):
        try:
            raw_data = fetch_stock_data(ticker_input, start_date, end_date)
            metadata = get_ticker_metadata(ticker_input)
        except Exception as exc:
            st.error(f"Data ingestion failed: {exc}")
            st.stop()

    if raw_data.empty:
        st.error(
            f"No historical records returned for ticker symbol '{ticker_input}'. "
            "Please check symbol spelling or broaden the date window."
        )
        st.stop()

    # Step 2: Validation & Feature Engineering
    try:
        valid_data = validate_timeseries_data(raw_data, lookback=lookback)
        enriched_data = compute_all_indicators(valid_data)
    except (InsufficientDataError, PreprocessingError) as exc:
        st.error(f"Preprocessing error: {exc}")
        st.stop()

    # Step 3: Top-level Metadata KPI Bar
    latest_close = enriched_data["Close"].iloc[-1]
    prev_close = enriched_data["Close"].iloc[-2] if len(enriched_data) > 1 else latest_close
    price_change = latest_close - prev_close
    pct_change = (price_change / prev_close) * 100.0

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Asset", ticker_input, metadata.get("shortName", ticker_input))
    kpi2.metric("Latest Close", f"${latest_close:,.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
    kpi3.metric("52-Wk High", f"${metadata.get('fiftyTwoWeekHigh', 0.0):,.2f}")
    kpi4.metric("52-Wk Low", f"${metadata.get('fiftyTwoWeekLow', 0.0):,.2f}")
    kpi5.metric("Total Sessions", f"{len(enriched_data):,}")

    st.markdown("---")

    # Step 4: Model Training with Live Feedback
    progress_bar = st.progress(0, text="Initializing Neural Network...")
    status_text = st.empty()

    def update_progress(curr: int, total: int, loss_val: float) -> None:
        pct = int((curr / total) * 100)
        progress_bar.progress(pct, text=f"Training Epoch {curr}/{total} — Loss: {loss_val:.5f}")

    try:
        results = train_and_predict(
            data=enriched_data,
            lookback=lookback,
            epochs=epochs,
            architecture=model_architecture,
            progress_callback=update_progress,
        )
        progress_bar.empty()
        status_text.empty()
    except Exception as exc:
        progress_bar.empty()
        st.error(f"Model training encountered a critical error: {exc}")
        st.stop()

    # Step 5: Tabbed Analytics Presentation
    tab_forecast, tab_technicals, tab_diagnostics, tab_registry = st.tabs(
        [
            "🔮 AI Forecast & Projections",
            "📊 Market Intelligence & Technicals",
            "📉 Model Diagnostics & Training",
            "📁 Data Registry & Export",
        ]
    )

    with tab_forecast:
        st.subheader("Held-Out Historical Test Evaluation")
        fig_eval = create_prediction_comparison_chart(results["table"], ticker_input)
        st.plotly_chart(fig_eval, use_container_width=True)

        st.subheader("Quantitative Performance Scorecard")
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("MAE ($)", f"{results['mae']:.2f}", help="Mean Absolute Error in dollar units")
        m_col2.metric("RMSE ($)", f"{results['rmse']:.2f}", help="Root Mean Squared Error penalizing large variances")
        m_col3.metric("MAPE (%)", f"{results['mape']:.2f}%", help="Mean Absolute Percentage Error")
        m_col4.metric("Directional Hit Ratio", f"{results['directional_accuracy']:.1f}%", help="Correct trend movement direction")
        m_col5.metric("R² Score", f"{results['r2']:.3f}", help="Coefficient of determination")

        st.markdown("---")
        st.subheader("Forward-Looking Autoregressive Projection")
        st.write(
            "Multi-step forward trajectory projected iteratively from the most recent historical trading sequence."
        )
        fig_future = create_future_forecast_chart(
            historical_df=enriched_data,
            forecast_df=results["future_forecast"],
            ticker=ticker_input,
            lookback_display=60,
        )
        st.plotly_chart(fig_future, use_container_width=True)
        st.dataframe(results["future_forecast"], use_container_width=True)

    with tab_technicals:
        st.subheader("Price Action & Volatility Bands")
        fig_candle = create_candlestick_chart(enriched_data, ticker_input)
        st.plotly_chart(fig_candle, use_container_width=True)

        st.subheader("Momentum & Trend Oscillators")
        fig_tech = create_technical_indicators_chart(enriched_data)
        st.plotly_chart(fig_tech, use_container_width=True)

    with tab_diagnostics:
        st.subheader("Optimization Trajectory (Loss Convergence)")
        fig_loss = create_loss_curve_chart(results["history"])
        st.plotly_chart(fig_loss, use_container_width=True)

        st.subheader("Model Architectural Specification")
        st.write(f"- **Architecture Selected:** `{model_architecture}`")
        st.write(f"- **Lookback Horizon:** `{lookback}` trading sessions")
        st.write(f"- **Optimization Algorithm:** `Adam(lr={config.model.learning_rate})`")
        st.write(f"- **Loss Criterion:** `Mean Squared Error (MSE)`")
        st.write(f"- **Regularization:** Dropout `{config.model.dropout_rate}` + EarlyStopping(patience={config.model.patience})")

    with tab_registry:
        st.subheader("Historical vs Predicted Evaluation Table")
        st.dataframe(results["table"], use_container_width=True)

        csv_bytes = results["table"].to_csv(index=True).encode("utf-8")
        st.download_button(
            label="📥 Download Test Predictions (CSV)",
            data=csv_bytes,
            file_name=f"{ticker_input.lower()}_forecast_evaluation.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    # Landing State
    st.info("👈 Configure market parameters in the sidebar and click **Ingest Data & Train Model** to start.")

    st.markdown(
        """
        ### System Architecture & Operational Blueprint
        **TrendPulse-AI** is an institutional-grade quantitative forecasting system designed for equities and indices.

        1. **Resilient Data Ingestion**: Pulls real-time and historical multi-year tick/daily quotes via yfinance with schema verification.
        2. **Leakage-Free Feature Engineering**: Normalizes time-series features strictly on training partitions to prevent lookahead bias.
        3. **Quantitative Signal Computation**: Ingests SMA (20, 50, 100, 200), EMA, RSI, MACD, and Bollinger Bands.
        4. **Deep Recurrent Forecasting**: Leverages Stacked LSTM, Bidirectional LSTM, and GRU networks with dynamic learning rate scheduling.
        5. **Interactive Visualization**: Generates real-time Plotly interactive financial dashboards with unified hover inspectors.
        """
    )
