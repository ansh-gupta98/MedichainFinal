"""
MediChain — FastAPI Backend
Entry point: loads InsightFace model once at startup for maximum speed.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import patient, doctor, health
from app.services.face_service import face_service
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy models once at startup, release on shutdown."""
    print("🚀 MediChain Backend starting...")
    face_service.load_model()
    print("✅ InsightFace ArcFace model loaded successfully.")
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
