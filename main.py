"""
MediChain — FastAPI Backend
Entry point: loads InsightFace model once at startup for maximum speed.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import patient, doctor, health
from app.services.face_service import face_service
from app.services.firebase_service import firebase_service
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:
      - Firebase: initialize now (lightweight — just reads credentials)
      - InsightFace: lazy load on first face request (heavy — 150MB model)
    """
    print("🚀 MediChain Backend starting...")

    # Firebase — initialize at startup (fast, no model download)
    try:
        firebase_service.initialize()
        print("✅ Firebase connected successfully.")
    except Exception as e:
        print(f"❌ Firebase init failed: {e}")
        print("   Check FIREBASE_CREDENTIALS_JSON env variable on Render.")

    # InsightFace — will load on first /doctor/identify or /patient/upload-photos call
    print("ℹ️  InsightFace will load on first face request (lazy loading).")
    print("✅ MediChain Backend ready!")

    yield
    print("🔴 MediChain Backend shutting down...")


app = FastAPI(
    title="MediChain API",
    description=(
        "AI-powered medical identity backend. "
        "Face recognition via InsightFace ArcFace R100, "
        "patient data from Firebase Firestore, "
        "AI summaries via Cerebras LLaMA."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the Kotlin / Flutter frontend + any local dev
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(health.router, tags=["Health"])
app.include_router(patient.router, prefix="/patient", tags=["Patient"])
app.include_router(doctor.router, prefix="/doctor", tags=["Doctor"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "project": "MediChain",
        "status": "running",
        "docs": "/docs",
        "version": "1.0.0",
    }
