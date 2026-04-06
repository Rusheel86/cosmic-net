# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU — use default index (handles ARM automatically)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cpu

# Install PyG core only — no C++ extensions needed for NNConv inference
RUN pip install --no-cache-dir torch-geometric==2.5.3

# Install rest of requirements
RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 2: lean runtime ─────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY app/ ./app/
COPY model_artifacts/ ./model_artifacts/
# Model file copied separately by CI pipeline

RUN useradd -m -u 1000 cosmicnet && chown -R cosmicnet /app
USER cosmicnet

EXPOSE 8000

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
