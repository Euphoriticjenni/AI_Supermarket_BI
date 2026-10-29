# AI_Supermarket_BI — Streamlit dashboard image
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build tools for any source-only wheels; curl for the container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so Docker layer caching skips reinstalls on code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Runtime directories (volumes from docker-compose mount over these).
RUN mkdir -p data/raw data/processed data/augmented models/registry logs reports database

EXPOSE 8501

# Streamlit's built-in health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
