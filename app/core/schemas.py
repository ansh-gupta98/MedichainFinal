"""
Pydantic schemas — request / response models for all API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Shared / Common
# ─────────────────────────────────────────────────────────────────────────────

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None


# ─────────────────────────────────────────────────────────────────────────────
# Patient — Registration & Photo Upload
# ─────────────────────────────────────────────────────────────────────────────

class PatientRegisterRequest(BaseModel):
    patient_uid: str = Field(..., description="Firebase Auth UID of the patient")
    name: str = Field(..., min_length=2, max_length=100)
    age: int = Field(..., ge=0, le=150)
    blood_group: str = Field(..., description="e.g. A+, B-, O+, AB+")
    allergies: List[str] = Field(default=[], description="List of known allergies")
    emergency_contact: Optional[str] = None


class PatientRegisterResponse(BaseModel):
    success: bool
    patient_uid: str
    message: str


class EmbeddingUploadResponse(BaseModel):
    success: bool
    patient_uid: str
    photos_processed: int
    photos_failed: int
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# Doctor — Face Identification
# ─────────────────────────────────────────────────────────────────────────────

class MedicalRecord(BaseModel):
    record_id: str
    date: Optional[str] = None
    hospital: Optional[str] = None
    doctor_name: Optional[str] = None
    diagnosis: Optional[str] = None
    prescription: Optional[str] = None
    report_urls: List[str] = []
    notes: Optional[str] = None


class PatientMedicalData(BaseModel):
    patient_uid: str
    name: str
    age: Optional[int] = None
    blood_group: Optional[str] = None
    allergies: List[str] = []
    emergency_contact: Optional[str] = None
    medical_records: List[MedicalRecord] = []
    embedding_updated_at: Optional[str] = None


class IdentifyResponse(BaseModel):
    success: bool
    matched: bool
    confidence_score: float = Field(
        ...,
        description="Cosine similarity score 0.0–1.0. Higher = more confident."
    )
    confidence_label: str = Field(
        ...,
        description="Human-readable label: HIGH / MEDIUM / LOW / NO_MATCH"
    )
    patient_data: Optional[PatientMedicalData] = None
    ai_summary: Optional[str] = Field(
        None,
        description="Cerebras AI-generated clinical summary for the doctor"
    )
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    insightface_loaded: bool
    firebase_connected: bool
    cerebras_configured: bool
    version: str = "1.0.0"
