import unittest
import json
from utils.narrative import visit_to_narrative, lab_to_narrative, patient_to_chunks


class TestNarrative(unittest.TestCase):
    def test_visit_to_narrative(self):
        visit = {
            "date": "2024-10-15",
            "doctor": "Dra. Martínez",
            "reason": "Routine checkup",
            "notes": "PA: 135/85",
            "diagnosis": "Controlled Hypertension",
        }
        res = visit_to_narrative(visit, "Juan Pérez")
        self.assertIn("Juan Pérez", res)
        self.assertIn("2024-10-15", res)
        self.assertIn("Dra. Martínez", res)
        self.assertIn("PA: 135/85", res)

    def test_lab_to_narrative(self):
        lab = {
            "date": "2024-10-20",
            "test_name": "HbA1c",
            "result": "7.2%",
            "reference_range": "4.0-5.6%",
            "interpretation": "High",
        }
        res = lab_to_narrative(lab, "Juan Pérez")
        self.assertIn("HbA1c", res)
        self.assertIn("7.2%", res)
        self.assertIn("High", res)

    def test_patient_to_chunks(self):
        patient_data = {
            "patient_id": "test-uuid",
            "demographics": {"name": "Test Patient"},
            "recent_visits": [{"visit_id": "v1", "date": "2024-01-01"}],
            "lab_results": [{"lab_id": "l1", "date": "2024-01-02"}],
        }
        chunks = patient_to_chunks(patient_data)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["metadata"]["event_type"], "visit")
        self.assertEqual(chunks[1]["metadata"]["event_type"], "lab")


if __name__ == "__main__":
    unittest.main()
