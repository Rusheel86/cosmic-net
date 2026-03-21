# Cosmic-Net API Dockerfile
# Multi-stage build for optimized image size

# =============================================================================
# Stage 1: Build environment
# =============================================================================
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch CPU version for smaller image (use GPU version for production with CUDA)
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric CPU
RUN pip install --no-cache-dir torch-geometric==2.4.0
RUN pip install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# Install other dependencies
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    scipy==1.11.3 \
    pandas==2.1.1 \
    scikit-learn==1.3.1 \
    fastapi==0.103.2 \
    uvicorn==0.23.2 \
    pydantic==2.4.2 \
    pyyaml==6.0.1 \
    python-dotenv==1.0.0 \
    requests==2.31.0 \
    h5py==3.9.0

# =============================================================================
# Stage 2: Runtime environment
# =============================================================================
FROM python:3.10-slim as runtime

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash cosmicnet
RUN chown -R cosmicnet:cosmicnet /app

# Copy application code
COPY --chown=cosmicnet:cosmicnet config/ ./config/
COPY --chown=cosmicnet:cosmicnet data/ ./data/
COPY --chown=cosmicnet:cosmicnet graph/ ./graph/
COPY --chown=cosmicnet:cosmicnet model/ ./model/
COPY --chown=cosmicnet:cosmicnet training/ ./training/
COPY --chown=cosmicnet:cosmicnet explain/ ./explain/
COPY --chown=cosmicnet:cosmicnet symbolic/ ./symbolic/
COPY --chown=cosmicnet:cosmicnet deploy/ ./deploy/
COPY --chown=cosmicnet:cosmicnet main.py .

# Create output directories
RUN mkdir -p outputs/checkpoints outputs/explanations outputs/equations outputs/logs \
    && chown -R cosmicnet:cosmicnet outputs/

# Switch to non-root user
USER cosmicnet

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command: start API server
CMD ["uvicorn", "deploy.api:app", "--host", "0.0.0.0", "--port", "8000"]
