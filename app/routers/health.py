"""
Health check router — used by Render to verify the service is alive.
Also reports status of all sub-systems: InsightFace, Firebase, Cerebras.
"""

from fastapi import APIRouter
from app.core.schemas import HealthResponse
from app.services.face_service import face_service
from app.services.firebase_service import firebase_service
from app.services.cerebras_service import cerebras_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Render pings this to verify the service is alive.
    """
    return HealthResponse(
        status="healthy",
        insightface_loaded=face_service.is_loaded,
        firebase_connected=firebase_service.is_connected,
        cerebras_configured=cerebras_service.is_configured,
        version="1.0.0",
    )
