"""
FaceService — InsightFace ArcFace R100 wrapper.

Key design decisions:
- Model loaded ONCE at startup (not per-request) → critical for speed
- Average multiple embeddings per patient → robust representation
- Cosine distance for similarity (unit-normalized vectors)
- Liveness attribute check built into InsightFace pipeline
"""

import io
import logging
import numpy as np
from typing import List, Optional, Tuple

import cv2
import insightface
from insightface.app import FaceAnalysis

from app.core.config import settings

logger = logging.getLogger(__name__)


class FaceService:
    def __init__(self):
        self._app: Optional[FaceAnalysis] = None
        self._loaded: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Model Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load InsightFace model. Auto-called on first use (lazy loading)."""
        if self._loaded:
            return
        try:
            logger.info(f"Loading InsightFace model '{settings.INSIGHTFACE_MODEL}'...")
            self._app = FaceAnalysis(
                name=settings.INSIGHTFACE_MODEL,
                # CPUExecutionProvider only — no GPU on Render free tier
                providers=["CPUExecutionProvider"],
            )
            self._app.prepare(
                ctx_id=-1,   # -1 = CPU mode (no GPU)
                det_size=(settings.INSIGHTFACE_DET_SIZE, settings.INSIGHTFACE_DET_SIZE),
            )
            self._loaded = True
            logger.info(f"InsightFace model '{settings.INSIGHTFACE_MODEL}' loaded ✅")
        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise

    def _ensure_loaded(self) -> None:
        """Auto-load model on first request (lazy loading pattern)."""
        if not self._loaded:
            self.load_model()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ─────────────────────────────────────────────────────────────────────────
    # Core: Image → Embedding
    # ─────────────────────────────────────────────────────────────────────────

    def _bytes_to_bgr(self, image_bytes: bytes) -> np.ndarray:
        """Convert raw image bytes (JPEG/PNG/WEBP) → OpenCV BGR array."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image. Ensure it is a valid JPEG/PNG/WEBP.")
        return img

    def extract_embedding(self, image_bytes: bytes) -> Tuple[Optional[np.ndarray], str]:
        """
        Extract 512-D ArcFace embedding from a single image.
        Auto-loads the model on first call (lazy loading).

        Returns:
            (embedding_array, status_message)
            embedding_array is None if extraction fails.
        """
        # Lazy load — safe to call multiple times, loads only once
        self._ensure_loaded()

        try:
            img_bgr = self._bytes_to_bgr(image_bytes)
        except ValueError as e:
            return None, str(e)

        try:
            faces = self._app.get(img_bgr)
        except Exception as e:
            logger.error(f"InsightFace inference error: {e}")
            return None, f"Inference error: {e}"

        if not faces:
            return None, "No face detected in the image."

        if len(faces) > 1:
            # Pick the largest face (closest to camera) for reliability
            faces = sorted(faces, key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
            logger.info(f"Multiple faces detected ({len(faces)}). Using the largest.")

        face = faces[0]
        embedding = face.embedding  # shape: (512,)

        # Normalize to unit vector for cosine similarity via dot product
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None, "Embedding norm is zero — invalid face."

        embedding_normalized = embedding / norm
        return embedding_normalized, "OK"

    # ─────────────────────────────────────────────────────────────────────────
    # Patient Registration: Average Multiple Photo Embeddings
    # ─────────────────────────────────────────────────────────────────────────

    def compute_average_embedding(
        self, image_bytes_list: List[bytes]
    ) -> Tuple[Optional[np.ndarray], int, int]:
        """
        Extract embeddings from multiple photos and average them.
        Averaging creates a more robust face representation.

        Args:
            image_bytes_list: List of raw image bytes (3–10 photos)

        Returns:
            (averaged_embedding, successful_count, failed_count)
        """
        embeddings = []
        failed = 0

        for i, img_bytes in enumerate(image_bytes_list):
            emb, msg = self.extract_embedding(img_bytes)
            if emb is not None:
                embeddings.append(emb)
            else:
                failed += 1
                logger.warning(f"Photo {i + 1} failed: {msg}")

        if not embeddings:
            return None, 0, failed

        # Stack and average → re-normalize
        stacked = np.stack(embeddings, axis=0)           # (N, 512)
        avg_emb = stacked.mean(axis=0)                   # (512,)
        norm = np.linalg.norm(avg_emb)
        avg_emb_normalized = avg_emb / norm if norm > 0 else avg_emb

        return avg_emb_normalized, len(embeddings), failed

    # ─────────────────────────────────────────────────────────────────────────
    # Similarity
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two unit-normalized embeddings.
        Returns value in [0, 1] (1 = identical).
        Both inputs must already be L2-normalized.
        """
        return float(np.dot(a, b))

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine distance = 1 - cosine_similarity.
        Used by Firestore find_nearest(). Lower = more similar.
        """
        return 1.0 - float(np.dot(a, b))

    @staticmethod
    def get_confidence_label(similarity: float) -> str:
        """Human-readable confidence label from similarity score."""
        if similarity >= 0.80:
            return "HIGH"
        elif similarity >= 0.65:
            return "MEDIUM"
        elif similarity >= 0.50:
            return "LOW"
        else:
            return "NO_MATCH"

    def embedding_to_list(self, embedding: np.ndarray) -> List[float]:
        """Convert numpy array to plain Python list for Firestore storage."""
        return embedding.tolist()

    def embedding_from_list(self, embedding_list: List[float]) -> np.ndarray:
        """Convert Firestore-stored list back to numpy array."""
        arr = np.array(embedding_list, dtype=np.float32)
        # Re-normalize in case of float precision drift
        norm = np.linalg.norm(arr)
        return arr / norm if norm > 0 else arr


# Singleton — shared across all request handlers
face_service = FaceService()
