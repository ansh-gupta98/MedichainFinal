# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — MediChain FastAPI Backend
# Optimized for Railway deployment (no RAM restrictions)
# Model: InsightFace buffalo_l (ArcFace R100) — BEST accuracy 99.83%
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Install system libs needed by OpenCV and InsightFace
# build-essential + g++ needed to compile InsightFace Cython extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libgomp1 \
    wget \
    build-essential \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download InsightFace buffalo_l model during build
# buffalo_l = ArcFace R100 — 99.83% accuracy — best model available
# Pre-downloading avoids cold-start delay on first request
RUN python -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=-1, det_size=(640, 640)); \
print('InsightFace buffalo_l (ArcFace R100) downloaded successfully.')"

# Copy application code
COPY . .

# Railway / Render both use PORT env variable
ENV PORT=8000

# Uvicorn with 2 workers for better throughput on Railway
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 2
