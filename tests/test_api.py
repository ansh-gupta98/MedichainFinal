"""
Quick integration tests — run with:
    pytest tests/ -v

These tests use httpx.AsyncClient to hit the FastAPI app without a real server.
Firebase and Cerebras calls are mocked so tests run offline.
"""

import io
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport

# Patch services before importing the app
with patch("app.services.face_service.FaceService.load_model"):
    from main import app


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_jpeg():
    """1x1 white JPEG for upload tests (real image bytes)."""
    from PIL import Image
    buf = io.BytesIO()
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "insightface_loaded" in data
    assert "firebase_connected" in data


@pytest.mark.asyncio
async def test_root():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert response.json()["project"] == "MediChain"


# ─────────────────────────────────────────────────────────────────────────────
# Patient registration
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_patient_register():
    with patch("app.services.firebase_service.firebase_service.initialize"), \
         patch("app.services.firebase_service.firebase_service.upsert_patient_profile"):

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/patient/register",
                json={
                    "patient_uid": "test_uid_001",
                    "name": "Rahul Sharma",
                    "age": 30,
                    "blood_group": "B+",
                    "allergies": ["Penicillin"],
                    "emergency_contact": "9876543210",
                },
            )

    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    assert data["patient_uid"] == "test_uid_001"


# ─────────────────────────────────────────────────────────────────────────────
# Photo upload — too few photos
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upload_photos_too_few(tiny_jpeg):
    with patch("app.services.face_service.face_service.is_loaded", True):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/patient/upload-photos",
                data={"patient_uid": "test_uid_001"},
                files=[
                    ("photos", ("p1.jpg", tiny_jpeg, "image/jpeg")),
                    ("photos", ("p2.jpg", tiny_jpeg, "image/jpeg")),
                ],
            )
    assert response.status_code == 422
    assert "Minimum 3 photos" in response.json()["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# Doctor identify — no face in photo
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_identify_no_face(tiny_jpeg):
    with patch("app.services.face_service.face_service.is_loaded", True), \
         patch(
             "app.services.face_service.face_service.extract_embedding",
             return_value=(None, "No face detected in the image."),
         ):

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/doctor/identify",
                files=[("photo", ("live.jpg", tiny_jpeg, "image/jpeg"))],
                data={"doctor_uid": "doc_001", "include_ai_summary": "false"},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["matched"] is False
    assert data["confidence_label"] == "NO_MATCH"


# ─────────────────────────────────────────────────────────────────────────────
# FaceService unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_cosine_similarity_identical():
    from app.services.face_service import FaceService
    a = np.array([1.0, 0.0, 0.0])
    assert FaceService.cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_orthogonal():
    from app.services.face_service import FaceService
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert FaceService.cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)


def test_confidence_label():
    from app.services.face_service import FaceService
    assert FaceService.get_confidence_label(0.90) == "HIGH"
    assert FaceService.get_confidence_label(0.70) == "MEDIUM"
    assert FaceService.get_confidence_label(0.55) == "LOW"
    assert FaceService.get_confidence_label(0.30) == "NO_MATCH"
