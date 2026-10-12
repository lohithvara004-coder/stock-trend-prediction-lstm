# 📈 TrendPulse-AI: Enterprise Stock Trend Prediction & Quantitative Analytics Engine

[![CI/CD Pipeline](https://github.com/lohithvara004-coder/stock-trend-prediction-lstm/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/lohithvara004-coder/stock-trend-prediction-lstm/actions)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F51B5.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Architected by [Lohith Varanasi](https://github.com/lohithvara004-coder)**  
> *Principal Software Architect & Quantitative Engineer Portfolio Project*

---

## 📌 1. Executive Summary & Problem Statement

Financial equity markets exhibit high stochasticity, non-stationarity, and nonlinear temporal dependencies that undermine classical autoregressive models (e.g., ARIMA). Standard retail machine learning implementations often suffer from:
1. **Lookahead Data Leakage:** Scaling entire time series prior to chronological partitioning.
2. **Static & Uninformative Visuals:** Static matplotlib plots that lack dynamic hover tooltips, volume subplots, and multi-asset indicators.
3. **Monolithic Spaghetti Architectures:** Single-script implementations without unit test suites, containerization, or structured error recovery.

**TrendPulse-AI** is an institutional-grade, full-stack quantitative analytics and deep learning forecasting platform. It bridges production software engineering with quantitative financial modeling:
- Ingests real-time and historical multi-year equities data via asynchronous market APIs.
- Applies mathematical feature engineering: **SMA (20/50/100/200), EMA (12/26), RSI (14), Bollinger Bands ($\pm 2\sigma$), and MACD**.
- Trains modular deep recurrent neural networks (**Stacked LSTM, Bidirectional LSTM, and Gated Recurrent Units - GRU**) with early stopping and learning rate scheduling.
- Provides **Zero-Leakage MinMax Scaling** and **Multi-Step Autoregressive Forward Trend Projections**.
- Ships with complete **PyTest suites (19 unit/integration tests)**, **Multi-Stage Docker builds**, and automated **GitHub Actions CI/CD workflows**.

---

## 🏛️ 2. System Architecture & Data Flow

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   CLIENT WEB LAYER                                      │
│                Streamlit Responsive Dashboard  +  Plotly Dynamic Canvas                 │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │ User Inputs (Ticker, Horizon, Model, Epochs)
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 CORE ENGINE (src/)                                      │
│                                                                                         │
│  ┌───────────────────────┐      ┌────────────────────────┐      ┌─────────────────────┐ │
│  │    data_handler.py    │ ───► │     indicators.py      │ ───► │  preprocessing.py   │ │
│  │ (Yahoo Finance API,   │      │ (SMA, EMA, RSI, MACD,  │      │ (Leakage-Free Norm, │ │
│  │  Metadata & Caching)  │      │  Bollinger Bands)      │      │  Sequence Windows)  │ │
│  └───────────────────────┘      └────────────────────────┘      └──────────┬──────────┘ │
│                                                                            │            │
│  ┌───────────────────────┐      ┌────────────────────────┐                 ▼            │
│  │   visualization.py    │ ◄─── │       metrics.py       │ ◄─── ┌─────────────────────┐ │
│  │ (Plotly Candlestick,  │      │ (MAE, RMSE, MAPE,      │      │      model.py       │ │
│  │  Loss Curves, Hover)  │      │  Directional Accuracy) │      │ (LSTM, BiLSTM, GRU, │ │
│  └───────────────────────┘      └────────────────────────┘      │  Autoregressive Proj│ │
│                                                                 └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### End-to-End Execution Lifecycle:
1. **Market Data Retrieval:** Ingests tick/daily OHLCV quotes using `yfinance` with boundary extension to guarantee inclusion of user-selected dates.
2. **Quality Assurance & Verification:** Validates chronological monotonicity, asserts absence of infinite values/NaNs, and checks for minimal sample thresholds.
3. **Quantitative Signal Computation:** Computes rolling momentum, trend, and volatility bands (SMA, EMA, RSI, Bollinger Bands, MACD).
4. **Chronological Splitting & Leakage-Free Scaling:** Divides dataset strictly into 80% train and 20% test. MinMax parameters are fitted solely on the training partition.
5. **3D Tensor Sequence Windowing:** Transforms 1D series into $(N, \text{lookback}, 1)$ tensors.
6. **Recurrent Neural Network Fitting:** Trains chosen architecture (Stacked LSTM, BiLSTM, or GRU) with `Adam` optimizer, `EarlyStopping(patience=4)`, and `ReduceLROnPlateau`.
7. **Held-Out Test Set Inference:** Computes predicted prices, inverts scaling to nominal dollar terms, and computes quantitative KPIs (MAE, RMSE, MAPE, Directional Hit Ratio, $R^2$).
8. **Autoregressive Forward Projection:** Iteratively feeds the latest predicted token back into the sliding window to project the next 7 business sessions.

---

## 📂 3. Directory & File Structure

```text
stock-trend-prediction-lstm/
├── .github/
│   └── workflows/
│       └── ci-cd.yml             # GitHub Actions CI/CD (Lint, Matrix Test, Audit, Docker)
├── docker/
│   ├── Dockerfile                # Production multi-stage build (builder + non-root runner)
│   └── docker-compose.yml        # Multi-container local orchestration & healthchecks
├── scripts/
│   ├── setup_env.ps1             # Automated virtual environment bootstrapper
│   └── run_dev.ps1               # Local development launch script
├── src/
│   ├── __init__.py               # Package metadata and version definition
│   ├── config.py                 # Centralized typed configuration & env overrides
│   ├── exceptions.py             # Custom domain exception hierarchy
│   ├── logger.py                 # Enterprise structured console logging
│   ├── data_handler.py           # Resilient market data ingestion & metadata fetching
│   ├── indicators.py             # Quantitative indicators (SMA, EMA, RSI, MACD, BB)
│   ├── preprocessing.py          # Leakage-free MinMax scaling & sequence creation
│   ├── model.py                  # Stacked LSTM, BiLSTM, GRU & autoregressive forecasting
│   ├── metrics.py                # Statistical error & directional accuracy metrics
│   └── visualization.py          # Interactive Plotly dashboards & candlestick charts
├── tests/
│   ├── __init__.py               # Test package initialization
│   ├── conftest.py               # PyTest fixtures (Geometric Brownian Motion generator)
│   ├── test_data_handler.py      # Symbol validation & API contract unit tests
│   ├── test_indicators.py        # Mathematical & geometric indicator tests
│   ├── test_preprocessing.py     # Sequence windowing & zero-leakage assertions
│   ├── test_model.py             # Architecture compilation & projection tests
│   └── test_metrics.py           # Statistical accuracy and error calculation tests
├── .dockerignore                 # Exclusions for lean container context
├── .gitignore                    # Enterprise git exclusions
├── app.py                        # Flagship Streamlit interactive web dashboard
├── Dockerfile                    # Root multi-stage Docker build
├── docker-compose.yml            # Root docker orchestration
├── LICENSE                       # MIT Open Source License
├── pytest.ini                    # PyTest configuration flags & warning filters
├── pyproject.toml                # Build system specification
├── requirements.txt              # Production application dependencies
├── requirements-dev.txt          # Developer, testing & linter dependencies
└── setup.cfg                     # Flake8 and MyPy configurations
```

---

## ⚡ 4. Quickstart Setup Guide

### Prerequisites
- Python 3.10 or 3.11 installed
- Git CLI
- Docker Desktop (optional, for containerized run)

### Local Native Installation

```bash
# 1. Clone repository
git clone https://github.com/lohithvara004-coder/stock-trend-prediction-lstm.git
cd stock-trend-prediction-lstm

# 2. Create and activate virtual environment
# Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install development dependencies
python -m pip install --upgrade pip
pip install -r requirements-dev.txt

# 4. Launch Streamlit Application
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🐳 5. Containerization & Docker Execution

### Multi-Stage Docker Build
Run isolated, unprivileged containerized instances with built-in healthchecks:

```bash
# Build production Docker image
docker build -t trendpulse-ai:latest .

# Run container exposing port 8501
docker run -d -p 8501:8501 --name trendpulse_app trendpulse-ai:latest

# Check health status
docker ps
```

### Docker Compose Orchestration

```bash
# Launch via Docker Compose with resource limits
docker-compose up -d --build

# Inspect container logs
docker-compose logs -f

# Shutdown services
docker-compose down
```

---

## 🧪 6. Testing, Quality Assurance & Linting

TrendPulse-AI enforces 100% test pass rate across mathematical, algorithmic, and data layers:

```bash
# Run complete test suite with verbose output
pytest -v

# Run test suite with coverage report
pytest --cov=src --cov-report=term-missing tests/

# Execute static code analysis
flake8 src/ tests/
black --check .
```

### Test Coverage Highlights:
- `test_indicators.py`: Ensures mathematical bounds on RSI ($0 \le \text{RSI} \le 100$), invariants of Bollinger Bands ($\text{Upper} \ge \text{Middle} \ge \text{Lower}$), and MACD signal convergence.
- `test_preprocessing.py`: Mathematically asserts zero lookahead leakage by verifying that `MinMaxScaler` parameters are calculated strictly on the training partition ($T \le \text{split\_index}$).
- `test_model.py`: Verifies tensor dimensionality contracts for Stacked LSTM, Bidirectional LSTM, and GRU, along with autoregressive forward projection bounds.
- `test_metrics.py`: Confirms statistical parity for MAE, RMSE, MAPE, $R^2$, and Directional Accuracy.

---

## 📊 7. Quantitative Metrics & Mathematical Formulations

| Metric | Formulation | Description |
| :--- | :--- | :--- |
| **MAE** | $\frac{1}{n} \sum_{i=1}^{n} \| y_i - \hat{y}_i \|$ | Mean Absolute Error in nominal dollar terms. |
| **RMSE** | $\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$ | Penalizes large outlier forecast errors. |
| **MAPE** | $\frac{100\%}{n} \sum_{i=1}^{n} \left\| \frac{y_i - \hat{y}_i}{y_i} \right\|$ | Mean Absolute Percentage Error. |
| **Hit Ratio (%)** | $\frac{1}{n-1} \sum_{i=2}^{n} \mathbb{I}(\text{sgn}(y_i - y_{i-1}) = \text{sgn}(\hat{y}_i - y_{i-1})) \times 100$ | Percentage of correct directional market moves. |
| **$R^2$ Score** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Coefficient of determination. |

---

## 🚀 8. Git & GitHub Deployment Execution Guide

Execute the following exact terminal commands to publish to your GitHub repository:

```bash
# 1. Initialize local Git repository
git init

# 2. Stage all production files
git add .

# 3. Create initial conventional commit
git commit -m "feat(core): initialize enterprise TrendPulse-AI stock forecasting platform"

# 4. Set default branch to main
git branch -M main

# 5. Connect remote repository
git remote add origin https://github.com/lohithvara004-coder/stock-trend-prediction-lstm.git

# 6. Push code to GitHub
git push -u origin main
```

---

## 📜 License & Author

Distributed under the **MIT License**. See `LICENSE` for more information.

**Author:** [Lohith Varanasi](https://github.com/lohithvara004-coder)  
**Profile:** [https://github.com/lohithvara004-coder](https://github.com/lohithvara004-coder)  
**Domain:** Quantitative Finance, Deep Learning, Enterprise Full-Stack AI Systems.
