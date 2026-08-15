# 🏥 MediChain — FastAPI Backend

> AI-powered medical identity system. Face recognition via **InsightFace ArcFace R100**, patient data from **Firebase Firestore**, AI clinical summaries via **Cerebras LLaMA 3.1 70B**.

---

## ⚡ Performance

| Step | Time |
|---|---|
| ArcFace R100 embedding (CPU) | ~150ms |
| Firestore KNN Vector Search | ~200ms |
| Patient data fetch | ~300ms |
| Cerebras AI summary | ~500ms |
| **Total end-to-end** | **~1.2s** ✅ |

Previous SVM approach took **10–15 minutes** because it was doing a full linear scan. Firestore KNN uses a vector index — it's O(log N) not O(N).

---

## 📁 Project Structure

```
medichain/
├── main.py                          # FastAPI app entrypoint
├── requirements.txt
├── Dockerfile
├── .env.example                     # Copy to .env and fill in
├── firestore.rules                  # Firebase security rules
├── firestore_vector_index.json      # Vector index definition
└── app/
    ├── core/
    │   ├── config.py                # All settings via env vars
    │   └── schemas.py               # Pydantic request/response models
    ├── services/
    │   ├── face_service.py          # InsightFace ArcFace wrapper
    │   ├── firebase_service.py      # Firestore KNN search + CRUD
    │   └── cerebras_service.py      # Cerebras LLaMA AI summary
    └── routers/
        ├── health.py                # GET /health
        ├── patient.py               # POST /patient/register, /upload-photos
        └── doctor.py                # POST /doctor/identify, /add-record
```

---

## 🚀 Local Setup

### 1. Clone & install

```bash
git clone https://github.com/your-org/medichain-backend
cd medichain-backend

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Firebase Service Account

1. Go to [Firebase Console](https://console.firebase.google.com/) → Your Project
2. **Project Settings** → **Service Accounts** → **Generate new private key**
3. Download the JSON file
4. Rename it to `firebase_credentials.json`
5. Place it in the project root (same folder as `main.py`)

> ⚠️ **Never commit this file to git.** It is already in `.gitignore`.

### 3. Create Firestore Vector Index

The KNN search requires a vector index. Create it via Firebase Console:

**Firebase Console → Firestore → Indexes → Add Index:**
- Collection: `patients`
- Field: `face_embedding`  
- Type: **Vector**
- Dimensions: `512`
- Query scope: Collection

**OR** via gcloud CLI:
```bash
gcloud firestore indexes composite create \
  --collection-group=patients \
  --query-scope=COLLECTION \
  --field-config=field-path=face_embedding,vector-config='{"dimension":"512","flat": {}}'
```

### 4. Environment Variables

```bash
cp .env.example .env
# Edit .env with your actual values
```

Required values in `.env`:
```
FIREBASE_CREDENTIALS_PATH=firebase_credentials.json
CEREBRAS_API_KEY=your-key-from-cloud.cerebras.ai
```

### 5. Run locally

```bash
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

---

## 🌐 Deploy to Render

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial MediChain backend"
git remote add origin https://github.com/your-org/medichain-backend.git
git push -u origin main
```

### Step 2: Create Render Web Service
1. Go to [render.com](https://render.com) → **New** → **Web Service**
2. Connect your GitHub repo
3. Set these settings:
   - **Environment**: Docker
   - **Region**: Any (pick closest to your users)
   - **Branch**: main
   - **Plan**: Free (for demo) or Starter (for production)

### Step 3: Add Environment Variables in Render
Go to your service → **Environment** tab and add:

| Key | Value |
|---|---|
| `APP_ENV` | `production` |
| `CEREBRAS_API_KEY` | your-cerebras-key |
| `FACE_MATCH_THRESHOLD` | `0.45` |
| `INSIGHTFACE_MODEL` | `buffalo_l` |

### Step 4: Add Firebase Credentials as Secret File
1. In Render → **Environment** → **Secret Files**
2. Add file: `/etc/secrets/firebase_credentials.json`
3. Paste your Firebase service account JSON content
4. Set env var: `FIREBASE_CREDENTIALS_PATH=/etc/secrets/firebase_credentials.json`

### Step 5: Deploy
Click **Deploy** — Render will build the Docker image and start the service.

> ℹ️ First deploy takes 5–10 minutes because Docker downloads the InsightFace model (~500MB). Subsequent deploys are fast due to layer caching.

**Your API URL:** `https://medichain-backend.onrender.com`

---

## 📡 API Reference

### `GET /health`
Check if all systems are running.

**Response:**
```json
{
  "status": "healthy",
  "insightface_loaded": true,
  "firebase_connected": true,
  "cerebras_configured": true,
  "version": "1.0.0"
}
```

---

### `POST /patient/register`
Register a new patient profile in Firestore.

**Request body (JSON):**
```json
{
  "patient_uid": "firebase_auth_uid_here",
  "name": "Rahul Sharma",
  "age": 30,
  "blood_group": "B+",
  "allergies": ["Penicillin", "Dust"],
  "emergency_contact": "9876543210"
}
```

**Response:**
```json
{
  "success": true,
  "patient_uid": "firebase_auth_uid_here",
  "message": "Patient 'Rahul Sharma' registered successfully."
}
```

---

### `POST /patient/upload-photos`
Upload 3–10 face photos to generate & store the ArcFace embedding.

**Request:** `multipart/form-data`
| Field | Type | Description |
|---|---|---|
| `patient_uid` | string (form) | Firebase Auth UID |
| `photos` | file[] (3–10) | JPEG/PNG/WEBP photos |

**Response:**
```json
{
  "success": true,
  "patient_uid": "firebase_auth_uid_here",
  "photos_processed": 7,
  "photos_failed": 1,
  "message": "Face embedding created from 7 photo(s) and stored successfully. 1 photo(s) were skipped (no face detected)."
}
```

---

### `POST /doctor/identify` ⭐ Main Endpoint
Identify a patient from a live photo. Returns full medical data + AI summary.

**Request:** `multipart/form-data`
| Field | Type | Description |
|---|---|---|
| `photo` | file | Live photo (JPEG/PNG/WEBP) |
| `doctor_uid` | string (optional) | Doctor's Firebase UID |
| `include_ai_summary` | bool (default: true) | Include Cerebras summary |

**Response (match found):**
```json
{
  "success": true,
  "matched": true,
  "confidence_score": 0.8734,
  "confidence_label": "HIGH",
  "patient_data": {
    "patient_uid": "abc123",
    "name": "Rahul Sharma",
    "age": 30,
    "blood_group": "B+",
    "allergies": ["Penicillin"],
    "emergency_contact": "9876543210",
    "medical_records": [
      {
        "record_id": "rec_001",
        "date": "2024-12-15",
        "hospital": "AIIMS Delhi",
        "doctor_name": "Dr. Priya Mehta",
        "diagnosis": "Type 2 Diabetes",
        "prescription": "Metformin 500mg twice daily",
        "report_urls": ["https://storage.firebase.../report.pdf"],
        "notes": "Follow up in 3 months"
      }
    ]
  },
  "ai_summary": "• **Critical Allergy:** Penicillin — avoid all β-lactam antibiotics\n• **Chronic Condition:** Type 2 Diabetes (diagnosed 2024) — on Metformin\n• **Blood Group:** B+\n• **Last visit:** AIIMS Delhi, Dec 2024\n• **Action needed:** Glucose monitoring, avoid penicillin-class drugs",
  "message": "Patient identified successfully with 87.3% confidence."
}
```

**Response (no match):**
```json
{
  "success": true,
  "matched": false,
  "confidence_score": 0.2341,
  "confidence_label": "NO_MATCH",
  "patient_data": null,
  "ai_summary": null,
  "message": "No confident match found. The patient may not be registered in MediChain."
}
```

---

### `POST /doctor/add-record`
Add a medical record for a patient after consultation.

**Request body (JSON):**
```json
{
  "patient_uid": "abc123",
  "date": "2025-08-11",
  "hospital": "Apollo Hospital",
  "doctor_name": "Dr. Suresh Kumar",
  "diagnosis": "Hypertension Stage 1",
  "prescription": "Amlodipine 5mg once daily",
  "report_urls": ["https://storage.firebase.../ecg_report.pdf"],
  "notes": "Lifestyle changes recommended. Review in 1 month."
}
```

---

## 🔐 Security Notes

- Face embeddings are stored as 512-D vectors — **raw photos are never stored on the server**
- Firebase service account key must never be committed to git
- All endpoints should be called with Firebase ID Token in `Authorization: Bearer <token>` header in production
- Firestore rules restrict patient data to the patient's own UID and verified doctors

---

## 🧪 Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## 🤝 Kotlin App Integration Example

```kotlin
// Retrofit call to identify patient
val multipartBody = MultipartBody.Builder()
    .setType(MultipartBody.FORM)
    .addFormDataPart(
        "photo", "live_capture.jpg",
        photoFile.asRequestBody("image/jpeg".toMediaType())
    )
    .addFormDataPart("doctor_uid", firebaseAuth.uid ?: "")
    .addFormDataPart("include_ai_summary", "true")
    .build()

val response = apiService.identifyPatient(multipartBody)
if (response.matched) {
    // Show patient data
    displayPatientData(response.patientData)
    displayAiSummary(response.aiSummary)
} else {
    showError("Patient not found: ${response.message}")
}
```

---

## 📞 Get Cerebras API Key

1. Visit [cloud.cerebras.ai](https://cloud.cerebras.ai/)
2. Sign up for free
3. Create an API key
4. Add to `.env` as `CEREBRAS_API_KEY=...`

The AI summary feature works without the key — it falls back to a structured text summary.
