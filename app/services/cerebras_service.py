"""
CerebrasService — AI-powered clinical summary generator.

Uses Cerebras LLaMA 3.1 70B API (2,100 tokens/sec) to generate
a concise, structured clinical summary for doctors after a patient
is identified. This helps doctors quickly understand the patient's
history without reading through all raw records.
"""

import logging
from typing import Dict, Any, List, Optional

from cerebras.cloud.sdk import Cerebras

from app.core.config import settings

logger = logging.getLogger(__name__)


class CerebrasService:
    def __init__(self):
        self._client: Optional[Cerebras] = None

    def _get_client(self) -> Cerebras:
        if self._client is None:
            if not settings.CEREBRAS_API_KEY:
                raise ValueError(
                    "CEREBRAS_API_KEY is not set. "
                    "Get your key from https://cloud.cerebras.ai/"
                )
            self._client = Cerebras(api_key=settings.CEREBRAS_API_KEY)
        return self._client

    @property
    def is_configured(self) -> bool:
        return bool(settings.CEREBRAS_API_KEY)

    # ─────────────────────────────────────────────────────────────────────────
    # Prompt Builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        patient_profile: Dict[str, Any],
        medical_records: List[Dict[str, Any]],
    ) -> str:
        """
        Build a structured prompt from raw Firestore patient data.
        Keeps it concise so Cerebras returns quickly.
        """
        name = patient_profile.get("name", "Unknown")
        age = patient_profile.get("age", "Unknown")
        blood_group = patient_profile.get("blood_group", "Unknown")
        allergies = patient_profile.get("allergies", [])
        emergency_contact = patient_profile.get("emergency_contact", "Not provided")

        allergies_str = ", ".join(allergies) if allergies else "None known"

        # Format medical records for the prompt
        records_text = ""
        if medical_records:
            for i, rec in enumerate(medical_records[:5], 1):  # max 5 for prompt size
                records_text += (
                    f"\n  Record {i}:"
                    f"\n    Date: {rec.get('date', 'Unknown')}"
                    f"\n    Hospital: {rec.get('hospital', 'Unknown')}"
                    f"\n    Diagnosis: {rec.get('diagnosis', 'Not specified')}"
                    f"\n    Prescription: {rec.get('prescription', 'None')}"
                    f"\n    Notes: {rec.get('notes', '')}"
                )
        else:
            records_text = "\n  No previous records found."

        prompt = f"""You are a clinical assistant AI helping a doctor quickly understand a patient's medical background.

PATIENT PROFILE:
- Name: {name}
- Age: {age}
- Blood Group: {blood_group}
- Known Allergies: {allergies_str}
- Emergency Contact: {emergency_contact}

MEDICAL HISTORY (most recent first):
{records_text}

TASK:
Generate a concise clinical summary (max 150 words) for the attending doctor. Include:
1. Key health risks and allergies (highlight critical ones)
2. Recent diagnoses and ongoing treatments
3. Any medication the patient is currently on
4. Important notes the doctor should know before treatment

Format as clear bullet points. Be direct and clinical. Do NOT include greetings or disclaimers."""

        return prompt

    # ─────────────────────────────────────────────────────────────────────────
    # Main: Generate Summary
    # ─────────────────────────────────────────────────────────────────────────

    def generate_medical_summary(
        self,
        patient_profile: Dict[str, Any],
        medical_records: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a clinical summary using Cerebras LLaMA 3.1 70B.

        Returns:
            Clinical summary string, or a fallback message on error.
        """
        if not self.is_configured:
            return self._fallback_summary(patient_profile, medical_records)

        try:
            client = self._get_client()
            prompt = self._build_prompt(patient_profile, medical_records)

            response = client.chat.completions.create(
                model=settings.CEREBRAS_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a concise clinical assistant. "
                            "Always respond with clear, structured bullet points. "
                            "Never exceed 150 words."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=300,
                temperature=0.3,   # Low temperature = factual, consistent output
            )

            summary = response.choices[0].message.content.strip()
            logger.info("Cerebras medical summary generated successfully.")
            return summary

        except Exception as e:
            logger.error(f"Cerebras API error: {e}")
            return self._fallback_summary(patient_profile, medical_records)

    # ─────────────────────────────────────────────────────────────────────────
    # Fallback (no API key or error)
    # ─────────────────────────────────────────────────────────────────────────

    def _fallback_summary(
        self,
        patient_profile: Dict[str, Any],
        medical_records: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a basic text summary without AI when Cerebras is unavailable.
        Ensures the API still returns useful data even without the AI key.
        """
        name = patient_profile.get("name", "Unknown")
        blood_group = patient_profile.get("blood_group", "Unknown")
        allergies = patient_profile.get("allergies", [])
        allergies_str = ", ".join(allergies) if allergies else "None known"
        record_count = len(medical_records)

        recent_diagnosis = ""
        if medical_records:
            latest = medical_records[0]
            recent_diagnosis = (
                f"• Most recent diagnosis: {latest.get('diagnosis', 'N/A')} "
                f"({latest.get('date', 'date unknown')})"
            )

        return (
            f"📋 Basic Medical Summary for {name}:\n"
            f"• Blood Group: {blood_group}\n"
            f"• Known Allergies: {allergies_str}\n"
            f"• Total Medical Records: {record_count}\n"
            f"{recent_diagnosis}\n"
            f"\n⚠️ AI summary unavailable — Cerebras API key not configured."
        )


# Singleton
cerebras_service = CerebrasService()
