"""
Application configuration via environment variables.
All secrets loaded from .env — never hardcoded.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # ── FastAPI ──────────────────────────────────────────────────────────────
    APP_ENV: str = "development"
    SECRET_KEY: str = "change-me-in-production"
    ALLOWED_ORIGINS: List[str] = ["*"]

    # ── InsightFace ───────────────────────────────────────────────────────────
    # buffalo_sc = ArcFace R50  (~150MB RAM) ← USE THIS on Render Free tier
    # buffalo_l  = ArcFace R100 (~500MB RAM) ← Use on Render Standard (2GB RAM)
    INSIGHTFACE_MODEL: str = "buffalo_sc"
    INSIGHTFACE_DET_SIZE: int = 640       # detection image size (px)
    FACE_MATCH_THRESHOLD: float = 0.45   # cosine distance threshold (lower = stricter)
    # Cosine distance range: 0 (identical) → 2 (opposite). 0.45 = ~99%+ confidence

    # ── Firebase ──────────────────────────────────────────────────────────────
    # Path to service-account JSON downloaded from Firebase console
    FIREBASE_CREDENTIALS_PATH: str = "firebase_credentials.json"
    FIRESTORE_PATIENTS_COLLECTION: str = "patients"

    # ── Cerebras ──────────────────────────────────────────────────────────────
    CEREBRAS_API_KEY: str = ""
    CEREBRAS_MODEL: str = "llama3.1-70b"  # 2100 tokens/sec — fastest available

    # ── Storage ───────────────────────────────────────────────────────────────
    # Temporary directory for uploaded photos (cleaned after processing)
    TEMP_UPLOAD_DIR: str = "/tmp/medichain_uploads"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
