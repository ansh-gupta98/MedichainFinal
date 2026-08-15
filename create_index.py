"""
create_index.py — Run this ONCE before deploying to create the Firestore Vector Index.

Usage:
    python create_index.py

This script creates the required vector index on patients.face_embedding
so that KNN search works. Run it only once per Firebase project.
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()


def get_access_token():
    """Get Google OAuth2 access token using service account credentials."""
    import google.auth
    import google.auth.transport.requests

    credentials_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase_credentials.json")

    # Load credentials from service account file
    with open(credentials_path, "r") as f:
        creds_data = json.load(f)

    project_id = creds_data.get("project_id")

    # Use google-auth to get access token
    from google.oauth2 import service_account

    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=scopes
    )

    # Refresh to get the token
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)

    return credentials.token, project_id


def create_vector_index():
    """Create Firestore vector index for face_embedding field."""

    print("🔄 Getting access token from Firebase service account...")
    try:
        token, project_id = get_access_token()
    except FileNotFoundError:
        print("❌ ERROR: firebase_credentials.json not found!")
        print("   → Download from Firebase Console → Project Settings → Service Accounts")
        return

    print(f"✅ Authenticated. Project ID: {project_id}")

    # Firestore REST API endpoint to create composite index
    url = (
        f"https://firestore.googleapis.com/v1/"
        f"projects/{project_id}/databases/(default)/collectionGroups/patients/indexes"
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Vector index definition — 512-D, flat (exact KNN, not approximate)
    index_body = {
        "queryScope": "COLLECTION",
        "fields": [
            {
                "fieldPath": "face_embedding",
                "vectorConfig": {
                    "dimension": 512,
                    "flat": {}      # flat = exact nearest neighbor search
                }
            }
        ]
    }

    print("🔄 Creating vector index on patients.face_embedding (dim=512)...")
    response = requests.post(url, headers=headers, json=index_body)

    if response.status_code == 200:
        data = response.json()
        print("✅ Vector index creation started!")
        print(f"   Operation: {data.get('name', 'N/A')}")
        print()
        print("⏳ Index creation takes 2–5 minutes to complete.")
        print("   Check status at: https://console.firebase.google.com")
        print("   Firestore → Indexes → you'll see it building.")

    elif response.status_code == 409:
        print("ℹ️  Index already exists! Nothing to do.")
        print("   Your KNN search is ready to use.")

    else:
        print(f"❌ Error creating index: {response.status_code}")
        print(f"   Response: {response.text}")
        print()
        print("💡 Try the gcloud CLI instead:")
        print(
            "   gcloud firestore indexes composite create \\\n"
            "     --collection-group=patients \\\n"
            "     --query-scope=COLLECTION \\\n"
            "     --field-config=field-path=face_embedding,"
            "vector-config='{\"dimension\":\"512\",\"flat\":{}}'"
        )


if __name__ == "__main__":
    create_vector_index()
