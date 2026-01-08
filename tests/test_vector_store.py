import unittest
from unittest.mock import MagicMock, patch
from core.vector_store import MedicalVectorStore
from qdrant_client.http import models


class TestVectorStore(unittest.TestCase):
    @patch("core.vector_store.QdrantClient")
    @patch("core.vector_store.voyageai.Client")
    def test_upsert_chunks(self, mock_voyage, mock_qdrant):
        # Setup mocks
        mock_voyage_instance = MagicMock()
        mock_voyage.return_value = mock_voyage_instance
        mock_voyage_instance.embed.return_value.embeddings = [[0.1] * 1024]

        mock_qdrant_instance = MagicMock()
        mock_qdrant.return_value = mock_qdrant_instance
        mock_qdrant_instance.get_collections.return_value.collections = []

        # Initialize store
        store = MedicalVectorStore()

        chunks = [
            {
                "text": "test narrative",
                "metadata": {
                    "patient_id": "p1",
                    "source_id": "s1",
                    "timestamp": "2024-01-01",
                    "event_type": "visit",
                },
            }
        ]

        store.upsert_chunks(chunks)

        # Verify qdrant upsert was called
        self.assertTrue(mock_qdrant_instance.upsert.called)
        call_args = mock_qdrant_instance.upsert.call_args
        self.assertEqual(call_args[1]["collection_name"], "medical_records")
        self.assertEqual(len(call_args[1]["points"]), 1)
        self.assertEqual(call_args[1]["points"][0].id, "p1_s1")


if __name__ == "__main__":
    unittest.main()
