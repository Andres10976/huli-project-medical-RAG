import os
import time
import json
import hashlib
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from core.vector_store import MedicalVectorStore
from utils.narrative import patient_to_chunks


class EHRWatcher(FileSystemEventHandler):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.vector_store = MedicalVectorStore()
        self.hashes = {}  # file_path -> content_hash
        self._initial_index()

    def _get_hash(self, file_path):
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _initial_index(self):
        print("Starting initial indexing...")
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.data_dir, filename)
                self._process_file(file_path)
        print("Initial indexing complete.")

    def _process_file(self, file_path):
        try:
            current_hash = self._get_hash(file_path)
            if self.hashes.get(file_path) == current_hash:
                return  # No change

            print(f"Processing changes in {file_path}...")
            with open(file_path, "r") as f:
                patient_data = json.load(f)

            chunks = patient_to_chunks(patient_data)
            self.vector_store.upsert_chunks(chunks)
            self.hashes[file_path] = current_hash
            print(f"Indexed {len(chunks)} chunks from {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".json"):
            self._process_file(event.src_path)

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".json"):
            self._process_file(event.src_path)


if __name__ == "__main__":
    DATA_DIRECTORY = "data"
    event_handler = EHRWatcher(DATA_DIRECTORY)
    observer = Observer()
    observer.schedule(event_handler, DATA_DIRECTORY, recursive=False)
    observer.start()
    print(f"Watching {DATA_DIRECTORY} for changes...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
