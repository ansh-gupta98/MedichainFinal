"""
FirebaseService — Firebase Admin SDK wrapper for Firestore operations.

Handles:
- Patient embedding storage (face_embedding field as array of 512 floats)
- KNN vector similarity search via Firestore find_nearest()
- Patient medical data retrieval (main doc + sub-collection records)
- Patient profile upsert
"""

import json
import logging
import os
import tempfile
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

from app.core.config import settings

logger = logging.getLogger(__name__)


class FirebaseService:
    def __init__(self):
        self._db = None
        self._initialized = False

    # ─────────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Initialize Firebase Admin SDK.
        Supports two ways to provide credentials (in priority order):

        1. FIREBASE_CREDENTIALS_JSON env var — full JSON string (Render recommended)
        2. FIREBASE_CREDENTIALS_PATH env var — path to JSON file (local dev)
        """
        if self._initialized:
            return
        try:
            cred = self._load_credentials()
            # Only initialize if not already done (handles hot-reload scenarios)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            self._db = firestore.client()
            self._initialized = True
            logger.info("Firebase Admin SDK initialized successfully.")
        except Exception as e:
            logger.error(f"Firebase initialization failed: {e}")
            raise

    def _load_credentials(self) -> credentials.Certificate:
        """
        Load Firebase credentials.

        Priority:
          1. FIREBASE_CREDENTIALS_JSON env var (JSON string) — used on Render
          2. FIREBASE_CREDENTIALS_PATH file path            — used locally
        """
        # ── Method 1: JSON string from environment variable (Render) ─────────
        creds_json_str = os.environ.get("FIREBASE_CREDENTIALS_JSON", "").strip()
        if creds_json_str:
            logger.info("Loading Firebase credentials from FIREBASE_CREDENTIALS_JSON env var.")
            try:
                creds_dict = json.loads(creds_json_str)
                return credentials.Certificate(creds_dict)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"FIREBASE_CREDENTIALS_JSON is not valid JSON: {e}\n"
                    "Make sure you pasted the entire firebase_credentials.json content."
                )

        # ── Method 2: File path (local development) ───────────────────────────
        file_path = settings.FIREBASE_CREDENTIALS_PATH
        if os.path.exists(file_path):
            logger.info(f"Loading Firebase credentials from file: {file_path}")
            return credentials.Certificate(file_path)

        raise FileNotFoundError(
            "Firebase credentials not found! Provide either:\n"
            "  • FIREBASE_CREDENTIALS_JSON env var (paste your JSON content), OR\n"
            f"  • A file at: {file_path}"
        )

    @property
    def db(self):
        if not self._initialized:
            self.initialize()
        return self._db

    @property
    def is_connected(self) -> bool:
        return self._initialized

    # ─────────────────────────────────────────────────────────────────────────
    # Patient Profile
    # ─────────────────────────────────────────────────────────────────────────

    def upsert_patient_profile(self, patient_uid: str, data: Dict[str, Any]) -> None:
        """
        Create or update a patient document in Firestore.
        Uses merge=True so existing fields are not overwritten.
        """
        ref = self.db.collection(settings.FIRESTORE_PATIENTS_COLLECTION).document(patient_uid)
        ref.set(data, merge=True)
        logger.info(f"Patient profile upserted: {patient_uid}")

    def get_patient_profile(self, patient_uid: str) -> Optional[Dict[str, Any]]:
        """Fetch a patient's main profile document."""
        ref = self.db.collection(settings.FIRESTORE_PATIENTS_COLLECTION).document(patient_uid)
        doc = ref.get()
        if doc.exists:
            return doc.to_dict()
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Embedding Storage
    # ─────────────────────────────────────────────────────────────────────────

    def store_patient_embedding(
        self,
        patient_uid: str,
        embedding: np.ndarray,
    ) -> None:
        """
        Store the averaged 512-D face embedding in Firestore as a Vector field.
        The Vector type enables Firestore's native KNN index.
        """
        from google.cloud.firestore_v1.vector import Vector
        from google.protobuf.timestamp_pb2 import Timestamp
        from datetime import datetime, timezone

        embedding_vector = Vector(embedding.tolist())

        ref = self.db.collection(settings.FIRESTORE_PATIENTS_COLLECTION).document(patient_uid)
        ref.set(
            {
                "face_embedding": embedding_vector,
                "embedding_updated_at": datetime.now(timezone.utc).isoformat(),
            },
            merge=True,
        )
        logger.info(f"Embedding stored for patient: {patient_uid}")

    # ─────────────────────────────────────────────────────────────────────────
    # KNN Vector Search — the fast matching core
    # ─────────────────────────────────────────────────────────────────────────

    def find_nearest_patient(
        self,
        query_embedding: np.ndarray,
        top_k: int = 1,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform KNN search on face_embedding field using Firestore Vector Search.

        Returns:
            List of (patient_uid, cosine_distance, patient_data_dict)
            Sorted by distance ascending (most similar first).

        Note:
            Requires a vector index on patients.face_embedding (dim=512, COSINE).
            Create via Firebase console or gcloud CLI — see README.
        """
        query_vector = Vector(query_embedding.tolist())

        collection_ref = self.db.collection(settings.FIRESTORE_PATIENTS_COLLECTION)

        # Firestore find_nearest performs server-side KNN — no full scan
        vector_query = collection_ref.find_nearest(
            vector_field="face_embedding",
            query_vector=query_vector,
            distance_measure=DistanceMeasure.COSINE,
            limit=top_k,
            distance_result_field="vector_distance",   # returned in each doc
        )

        results = []
        docs = vector_query.get()

        for doc in docs:
            data = doc.to_dict()
            distance = data.pop("vector_distance", 1.0)   # cosine distance [0,2]
            # Convert cosine distance [0,2] → similarity [0,1]
            # For unit vectors: similarity = 1 - distance (distance ∈ [0,1])
            similarity = max(0.0, 1.0 - float(distance))
            results.append((doc.id, similarity, data))

        logger.info(
            f"Vector search returned {len(results)} result(s). "
            + (f"Top similarity: {results[0][1]:.4f}" if results else "No results.")
        )
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Medical Records Sub-collection
    # ─────────────────────────────────────────────────────────────────────────

    def get_medical_records(self, patient_uid: str) -> List[Dict[str, Any]]:
        """
        Fetch all medical records for a patient from the
        patients/{uid}/medical_records sub-collection.
        Ordered by date descending (most recent first).
        """
        records_ref = (
            self.db.collection(settings.FIRESTORE_PATIENTS_COLLECTION)
            .document(patient_uid)
            .collection("medical_records")
            .order_by("date", direction=firestore.Query.DESCENDING)
        )

        docs = records_ref.stream()
        records = []
        for doc in docs:
            record = doc.to_dict()
            record["record_id"] = doc.id
            records.append(record)

        logger.info(f"Fetched {len(records)} medical record(s) for patient: {patient_uid}")
        return records

    def add_medical_record(
        self, patient_uid: str, record_data: Dict[str, Any]
    ) -> str:
        """Add a new medical record to a patient's sub-collection. Returns record ID."""
        ref = (
            self.db.collection(settings.FIRESTORE_PATIENTS_COLLECTION)
            .document(patient_uid)
            .collection("medical_records")
            .document()
        )
        ref.set(record_data)
        logger.info(f"Medical record added: {ref.id} for patient: {patient_uid}")
        return ref.id


# Singleton instance
firebase_service = FirebaseService()
