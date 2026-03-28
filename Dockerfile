# ── Auto-SWE OpenEnv Environment ──────────────────────────────────────────
# Optimised for Hugging Face Spaces (port 7860, 2 vCPU / 8 GB RAM)

FROM python:3.11-slim

# Avoid interactive prompts & set UTF-8
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency spec first for layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . /app/

# Set PYTHONPATH so our modules can be imported natively
ENV PYTHONPATH="/app:$PYTHONPATH"

# Expose HF Spaces port
EXPOSE 7860

# Run the FastAPI server directly through Python to avoid uvicorn resolution issues
CMD ["python", "server/app.py"]
