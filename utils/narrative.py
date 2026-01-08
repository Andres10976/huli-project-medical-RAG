import json


def visit_to_narrative(visit, patient_name):
    """Structured text for visits: DATE | DOCTOR | REASON | NOTES"""
    date = visit.get("date", "N/A")
    doctor = visit.get("doctor", "N/A")
    reason = visit.get("reason", "N/A")
    notes = visit.get("notes", "N/A")

    return f"DATE: {date} | DOCTOR: {doctor} | REASON: {reason} | NOTES: {notes}"


def lab_to_narrative(lab, patient_name):
    """Structured text for labs: DATE | TEST | RESULTS"""
    date = lab.get("date", "N/A")
    test_name = lab.get("test") or lab.get("test_name") or "N/A"

    parts = [f"DATE: {date}", f"TEST: {test_name}"]

    results = lab.get("results")
    if isinstance(results, dict):
        for k, v in results.items():
            parts.append(f"{k}: {v}")
    else:
        res = lab.get("result") or lab.get("value") or "N/A"
        parts.append(f"RESULT: {res}")

    return " | ".join(parts)


def clean_metadata(metadata):
    """Removes None or empty string values from metadata."""
    return {k: v for k, v in metadata.items() if v is not None and v != ""}


def patient_to_chunks(patient_data):
    """
    Splits patient data into atomic units for vector indexing.
    Structured text format (not narrative).
    """
    patient_id = patient_data.get("patient_id")
    chunks = []

    # Process visits
    for i, visit in enumerate(patient_data.get("recent_visits", [])):
        text = visit_to_narrative(visit, None)
        metadata = clean_metadata(
            {
                "patient_id": patient_id,
                "timestamp": visit.get("date"),
                "event_type": "visit",
                "doctor": visit.get("doctor"),
            }
        )
        internal_id = visit.get("visit_id") or f"v{i}"
        chunks.append({"text": text, "metadata": metadata, "internal_id": internal_id})

    # Process labs
    for i, lab in enumerate(patient_data.get("lab_results", [])):
        text = lab_to_narrative(lab, None)
        metadata = clean_metadata(
            {
                "patient_id": patient_id,
                "timestamp": lab.get("date"),
                "event_type": "lab",
                "test_name": lab.get("test") or lab.get("test_name"),
            }
        )
        internal_id = lab.get("lab_id") or f"l{i}"
        chunks.append({"text": text, "metadata": metadata, "internal_id": internal_id})

    return chunks
