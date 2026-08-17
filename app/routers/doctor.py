"""
Doctor router — the core identification pipeline.

POST /doctor/identify
  1. Receive a live photo from the doctor's app
  2. Extract ArcFace embedding
  3. Firestore KNN vector search → find nearest patient
  4. Threshold check (reject if similarity too low)
  5. Fetch full patient profile + medical records
  6. Generate Cerebras AI clinical summary
  7. Return everything to the doctor in < 5 seconds

POST /doctor/add-record
  Allows the doctor to add a new medical record for a patient
  after identification.
"""

import logging

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, status, Body
from typing import Optional, Dict, Any

from app.core.schemas import (
    IdentifyResponse,
    PatientMedicalData,
    MedicalRecord,
    APIResponse,
)
from app.services.face_service import face_service
from app.services.firebase_service import firebase_service
from app.services.cerebras_service import cerebras_service
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# POST /doctor/identify  ← The main endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/identify",
    response_model=IdentifyResponse,
    summary="Identify patient from live photo",
    description="""
**Core MediChain endpoint.**

The doctor captures a live photo of the patient.
This endpoint:
1. Extracts a 512-D ArcFace R100 embedding from the photo
2. Runs Firestore KNN Vector Search to find the closest match
3. If similarity ≥ threshold (default 0.50), fetches the patient's full medical data
4. Generates a Cerebras AI clinical summary for quick doctor review
5. Returns everything in a single response

**Expected response time:** 1–3 seconds on Render standard tier.
    """,
)
async def identify_patient(
    photo: UploadFile = File(
        ...,
        description="Live photo of patient (JPEG/PNG/WEBP). Single clear face required.",
    ),
    doctor_uid: Optional[str] = Form(
        None,
        description="Firebase UID of the doctor making the request (for audit logging).",
    ),
    include_ai_summary: bool = Form(
        True,
        description="Set false to skip Cerebras AI summary (faster response).",
    ),
):
    # ── Validate photo type ───────────────────────────────────────────────────
    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if photo.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{photo.content_type}'. Use JPEG, PNG, or WEBP.",
        )

    # ── Step 1: Read photo bytes ──────────────────────────────────────────────
    photo_bytes = await photo.read()
    logger.info(
        f"Identify request from doctor '{doctor_uid}' — "
        f"photo size: {len(photo_bytes) / 1024:.1f} KB"
    )

    # ── Step 2: Extract ArcFace embedding ────────────────────────────────────
    query_embedding, msg = face_service.extract_embedding(photo_bytes)

    if query_embedding is None:
        return IdentifyResponse(
            success=False,
            matched=False,
            confidence_score=0.0,
            confidence_label="NO_MATCH",
            patient_data=None,
            ai_summary=None,
            message=f"Face extraction failed: {msg}. Ensure the photo shows a clear, single face.",
        )

    # ── Step 3: Firestore KNN Vector Search ───────────────────────────────────
    try:
        firebase_service.initialize()
        results = firebase_service.find_nearest_patient(query_embedding, top_k=1)
    except Exception as e:
        logger.error(f"Firestore KNN search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database search failed: {str(e)}",
        )

    if not results:
        return IdentifyResponse(
            success=True,
            matched=False,
            confidence_score=0.0,
            confidence_label="NO_MATCH",
            patient_data=None,
            ai_summary=None,
            message="No patients registered in the system yet.",
        )

    patient_uid, similarity, raw_data = results[0]
    confidence_label = face_service.get_confidence_label(similarity)

    logger.info(
        f"KNN match: patient_uid={patient_uid}, "
        f"similarity={similarity:.4f}, label={confidence_label}"
    )

    # ── Step 4: Threshold check ───────────────────────────────────────────────
    # Similarity threshold: reject matches below 0.50 to prevent false positives
    # Medical context — we set a conservative threshold
    match_threshold = 1.0 - settings.FACE_MATCH_THRESHOLD  # convert distance→similarity

    if similarity < match_threshold:
        return IdentifyResponse(
            success=True,
            matched=False,
            confidence_score=round(similarity, 4),
            confidence_label="NO_MATCH",
            patient_data=None,
            ai_summary=None,
            message=(
                f"No confident match found (similarity={similarity:.2%}). "
                "The patient may not be registered in MediChain."
            ),
        )

    # ── Step 5: Fetch full patient profile + medical records ─────────────────
    try:
        profile = firebase_service.get_patient_profile(patient_uid)
        if not profile:
            logger.warning(f"Patient {patient_uid} has embedding but no profile doc!")
            profile = {}

        records_raw = firebase_service.get_medical_records(patient_uid)
    except Exception as e:
        logger.error(f"Patient data fetch error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch patient data: {str(e)}",
        )

    # ── Build structured patient data ────────────────────────────────────────
    medical_records = [
        MedicalRecord(
            record_id=rec.get("record_id", ""),
            date=str(rec.get("date", "")) if rec.get("date") else None,
            hospital=rec.get("hospital"),
            doctor_name=rec.get("doctor_name") or rec.get("doctor"),
            diagnosis=rec.get("diagnosis"),
            prescription=rec.get("prescription"),
            report_urls=rec.get("report_urls", []),
            notes=rec.get("notes"),
        )
        for rec in records_raw
    ]

    patient_data = PatientMedicalData(
        patient_uid=patient_uid,
        name=profile.get("name", "Unknown"),
        age=profile.get("age"),
        blood_group=profile.get("blood_group"),
        allergies=profile.get("allergies", []),
        emergency_contact=profile.get("emergency_contact"),
        medical_records=medical_records,
        embedding_updated_at=profile.get("embedding_updated_at"),
    )

    # ── Step 6: Cerebras AI clinical summary ─────────────────────────────────
    ai_summary = None
    if include_ai_summary:
        try:
            ai_summary = cerebras_service.generate_medical_summary(profile, records_raw)
        except Exception as e:
            logger.warning(f"Cerebras summary failed (non-critical): {e}")
            ai_summary = "AI summary temporarily unavailable."

    # ── Step 7: Return response ───────────────────────────────────────────────
    return IdentifyResponse(
        success=True,
        matched=True,
        confidence_score=round(similarity, 4),
        confidence_label=confidence_label,
        patient_data=patient_data,
        ai_summary=ai_summary,
        message=f"Patient identified successfully with {similarity:.1%} confidence.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /doctor/add-record
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/add-record",
    response_model=APIResponse,
    summary="Add a medical record for a patient (post-consultation)",
    description=(
        "Allows a doctor to add a new medical record after identifying the patient. "
        "Record is stored in patients/{uid}/medical_records sub-collection."
    ),
)
async def add_medical_record(
    patient_uid: str = Body(..., embed=True),
    date: Optional[str] = Body(None, embed=True),
    hospital: Optional[str] = Body(None, embed=True),
    doctor_name: Optional[str] = Body(None, embed=True),
    diagnosis: Optional[str] = Body(None, embed=True),
    prescription: Optional[str] = Body(None, embed=True),
    report_urls: Optional[list] = Body(default=[], embed=True),
    notes: Optional[str] = Body(None, embed=True),
):
    try:
        firebase_service.initialize()

        record_data = {
            "date": date,
            "hospital": hospital,
            "doctor_name": doctor_name,
            "diagnosis": diagnosis,
            "prescription": prescription,
            "report_urls": report_urls or [],
            "notes": notes,
        }
        # Remove None values
        record_data = {k: v for k, v in record_data.items() if v is not None}

        record_id = firebase_service.add_medical_record(patient_uid, record_data)

        return APIResponse(
            success=True,
            message=f"Medical record added successfully.",
            data={"record_id": record_id, "patient_uid": patient_uid},
        )
    except Exception as e:
        logger.error(f"Add record error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add record: {str(e)}",
        )
