"""
Patient router — handles patient registration and photo/embedding upload.

Endpoints:
  POST /patient/register      — Save patient profile to Firestore
  POST /patient/upload-photos — Accept 3–10 photos, generate averaged
                                ArcFace embedding, store in Firestore
  GET  /patient/{uid}         — Retrieve patient profile
"""

import logging
from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, status
from fastapi.responses import JSONResponse

from app.core.schemas import (
    PatientRegisterRequest,
    PatientRegisterResponse,
    EmbeddingUploadResponse,
    APIResponse,
)
from app.services.face_service import face_service
from app.services.firebase_service import firebase_service
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# POST /patient/register
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=PatientRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new patient profile",
    description=(
        "Creates or updates a patient document in Firestore with basic "
        "profile information. Call this BEFORE uploading photos."
    ),
)
async def register_patient(request: PatientRegisterRequest):
    try:
        firebase_service.initialize()

        profile_data = {
            "name": request.name,
            "age": request.age,
            "blood_group": request.blood_group,
            "allergies": request.allergies,
            "emergency_contact": request.emergency_contact,
        }

        firebase_service.upsert_patient_profile(request.patient_uid, profile_data)

        return PatientRegisterResponse(
            success=True,
            patient_uid=request.patient_uid,
            message=f"Patient '{request.name}' registered successfully.",
        )

    except Exception as e:
        logger.error(f"Patient registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# POST /patient/upload-photos
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/upload-photos",
    response_model=EmbeddingUploadResponse,
    summary="Upload 3–10 photos to generate face embedding",
    description=(
        "Accepts 3 to 10 face photos (JPEG/PNG/WEBP). "
        "Extracts ArcFace R100 embeddings from each, averages them "
        "for a robust representation, and stores the result in Firestore "
        "as a 512-dimensional vector. "
        "The patient must be registered first via /patient/register."
    ),
)
async def upload_patient_photos(
    patient_uid: str = Form(..., description="Firebase Auth UID of the patient"),
    photos: List[UploadFile] = File(
        ...,
        description="3 to 10 face photos (JPEG/PNG/WEBP). More = more accurate.",
    ),
):
    # ── Validation ────────────────────────────────────────────────────────────
    if len(photos) < 3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Minimum 3 photos required. Got {len(photos)}.",
        )
    if len(photos) > 10:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Maximum 10 photos allowed. Got {len(photos)}.",
        )

    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    for photo in photos:
        if photo.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    f"File '{photo.filename}' has unsupported type '{photo.content_type}'. "
                    f"Allowed: JPEG, PNG, WEBP."
                ),
            )

    if not face_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition model is not ready. Try again shortly.",
        )

    # ── Read all photo bytes ──────────────────────────────────────────────────
    image_bytes_list = []
    for photo in photos:
        content = await photo.read()
        image_bytes_list.append(content)

    # ── Generate averaged ArcFace embedding ───────────────────────────────────
    logger.info(
        f"Generating embedding for patient {patient_uid} "
        f"from {len(image_bytes_list)} photos..."
    )
    avg_embedding, success_count, fail_count = face_service.compute_average_embedding(
        image_bytes_list
    )

    if avg_embedding is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Could not detect a face in any of the {len(photos)} uploaded photos. "
                "Ensure photos are clear, well-lit, and show a single face."
            ),
        )

    if success_count < 3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Only {success_count} out of {len(photos)} photos had a detectable face. "
                "At least 3 valid face photos are required."
            ),
        )

    # ── Store in Firestore ────────────────────────────────────────────────────
    try:
        firebase_service.initialize()
        firebase_service.store_patient_embedding(patient_uid, avg_embedding)
    except Exception as e:
        logger.error(f"Firestore embedding storage error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store embedding: {str(e)}",
        )

    logger.info(
        f"Embedding stored for {patient_uid}: "
        f"{success_count} succeeded, {fail_count} failed."
    )

    return EmbeddingUploadResponse(
        success=True,
        patient_uid=patient_uid,
        photos_processed=success_count,
        photos_failed=fail_count,
        message=(
            f"Face embedding created from {success_count} photo(s) "
            f"and stored successfully. "
            + (f"{fail_count} photo(s) were skipped (no face detected)." if fail_count else "")
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /patient/{uid}
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{patient_uid}",
    response_model=APIResponse,
    summary="Get patient profile by UID",
)
async def get_patient(patient_uid: str):
    try:
        firebase_service.initialize()
        profile = firebase_service.get_patient_profile(patient_uid)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No patient found with UID: {patient_uid}",
            )
        # Remove embedding from response (not useful to clients, very large)
        profile.pop("face_embedding", None)

        return APIResponse(success=True, message="Patient found.", data=profile)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get patient error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
