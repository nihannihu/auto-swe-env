# ── Auto-SWE OpenEnv Environment ──────────────────────────────────────────
# Optimised for Hugging Face Spaces (port 7860, 2 vCPU / 8 GB RAM)

FROM python:3.11-slim

# Avoid interactive prompts & set UTF-8
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8

# Create a non-root user matching the Hugging Face standard ID 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency spec first for layer caching
COPY requirements.txt ./

# Install Python dependencies natively
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source into the workspace
COPY . .

# IMPORTANT: Grant complete permission to the User 1000 so the agent
# has rights to write physical Python bug files and run Pytest dynamically.
RUN chown -R 1000:1000 $HOME/app

# Switch to the non-root worker user
USER user

# Set PYTHONPATH so our modules can be imported natively
ENV PYTHONPATH="$HOME/app:$PYTHONPATH"

# Expose HF Spaces port
EXPOSE 7860

# Run the FastAPI server directly through Python to avoid uvicorn resolution issues
CMD ["python", "server/app.py"]
