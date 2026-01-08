import os
import json
import sys

# Add project root to sys.path to resolve core and utils modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vector_store import MedicalVectorStore
from utils.narrative import patient_to_chunks
from dotenv import load_dotenv

load_dotenv()


def first_run_ingestion():
    print("🚀 Starting First Run Ingestion...")
    vs = MedicalVectorStore()
    data_dir = "data"

    # Check if we need to recreate the collection due to dimension change
    print(f"Checking collection '{vs.collection_name}'...")
    try:
        vs.client.get_collection(vs.collection_name)
        print(
            f"Collection exists. Recreating it to ensure 512-dimension configuration..."
        )
        vs.client.delete_collection(vs.collection_name)
    except Exception:
        print("Collection does not exist. Creating new one...")

    vs._ensure_collection()

    files = [
        f for f in os.listdir(data_dir) if f.endswith(".json") and f != "example.json"
    ]

    if not files:
        print("❌ No patient files found in data/ directory.")
        return

    for filename in files:
        file_path = os.path.join(data_dir, filename)
        print(f"📄 Processing {filename}...")

        try:
            with open(file_path, "r") as f:
                patient_data = json.load(f)

            chunks = patient_to_chunks(patient_data)
            vs.upsert_chunks(chunks)
            print(
                f"✅ Ingested {len(chunks)} chunks for {patient_data['demographics']['name']}"
            )
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print("\n✨ First run ingestion complete! Your Vector DB is ready.")


if __name__ == "__main__":
    first_run_ingestion()
