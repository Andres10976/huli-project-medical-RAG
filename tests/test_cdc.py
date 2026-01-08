import unittest
from unittest.mock import MagicMock, patch
import os
import json
import shutil
from cdc.watcher import EHRWatcher


class TestCDCWatcher(unittest.TestCase):
    def setUp(self):
        self.test_data_dir = "test_data_cdc"
        os.makedirs(self.test_data_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_data_dir)

    @patch("cdc.watcher.MedicalVectorStore")
    def test_watcher_initial_index(self, mock_store):
        # Create a test file
        test_file = os.path.join(self.test_data_dir, "test_patient.json")
        patient_data = {
            "patient_id": "p1",
            "demographics": {"name": "Test"},
            "recent_visits": [{"visit_id": "v1", "date": "2024-01-01"}],
        }
        with open(test_file, "w") as f:
            json.dump(patient_data, f)

        # Initialize watcher
        watcher = EHRWatcher(self.test_data_dir)

        # Verify vector store upsert was called during initial indexing
        self.assertTrue(mock_store.return_value.upsert_chunks.called)


if __name__ == "__main__":
    unittest.main()
