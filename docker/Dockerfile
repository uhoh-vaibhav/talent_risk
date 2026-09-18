# Multi-stage production Dockerfile for Talent Risk Scoring System API
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy codebase
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
