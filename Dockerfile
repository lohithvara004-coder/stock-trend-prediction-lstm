# ==============================================================================
# Multi-Stage Production Dockerfile for TrendPulse-AI
# Target: Python 3.11-slim base for minimal footprint and security hardening
# ==============================================================================

# Stage 1: Build Dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install compiler dependencies required for certain scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Compile wheels into local cache wheelhouse
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# ==============================================================================
# Stage 2: Minimal Distroless/Slim Production Runtime
# ==============================================================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Install minimal curl for docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged system user for hardened container execution
RUN groupadd -r appgroup && useradd -r -g appgroup -u 10001 appuser

# Copy pre-built wheels from builder stage
COPY --from=builder /build/wheels /wheels
COPY requirements.txt .

# Install dependencies from pre-compiled wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Copy application source code
COPY --chown=appuser:appgroup app.py .
COPY --chown=appuser:appgroup src/ ./src/

# Switch to non-root user
USER appuser

# Expose standard Streamlit port
EXPOSE 8501

# Healthcheck to verify web service availability
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Environment flags for optimal container execution
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

ENTRYPOINT ["streamlit", "run", "app.py"]
