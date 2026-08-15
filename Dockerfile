# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — MediChain FastAPI Backend
# Optimized for Render deployment
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Install system libs needed by OpenCV and InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download InsightFace buffalo_l model during build
# This avoids cold-start delay on first request
RUN python -c "\
import insightface; \
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=-1, det_size=(640, 640)); \
print('InsightFace buffalo_l model downloaded successfully.')"

# Copy application code
COPY . .

# Render uses PORT env variable
ENV PORT=8000

# Run with uvicorn — single worker on Render free tier
# Increase workers on paid tier for higher throughput
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1
