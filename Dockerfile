# ── Auto-SWE OpenEnv Environment ──────────────────────────────────────────
# Optimised for Hugging Face Spaces (port 7860, 2 vCPU / 8 GB RAM)
# Build:  docker build -t auto-swe-env .
# Run:    docker run -p 7860:7860 auto-swe-env

FROM python:3.11-slim AS builder

# Avoid interactive prompts & set UTF-8
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8

WORKDIR /app

# Install system dependencies (git for pip installs, gcc for potential C deps)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency spec first for layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
COPY . /app/

# Set PYTHONPATH so our modules can be imported
ENV PYTHONPATH="/app:$PYTHONPATH"

# Expose HF Spaces port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
