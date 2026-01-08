import json


def visit_to_narrative(visit, patient_name):
    """Converts a visit JSON object to a natural language narrative."""
    date = visit.get("date", "Unknown date")
    doctor = visit.get("doctor", "a healthcare provider")
    reason = visit.get("reason", "a consultation")
    notes = visit.get("notes", "No specific notes recorded.")
    diagnosis = visit.get("diagnosis", "No formal diagnosis mentioned.")

    narrative = (
        f"On {date}, the patient {patient_name} had a {reason} with {doctor}. "
        f"Clinical notes: {notes}. "
        f"Diagnosis/Outcome: {diagnosis}."
    )
    return narrative


def lab_to_narrative(lab, patient_name):
    """Converts a lab result JSON object to a natural language narrative."""
    date = lab.get("date", "Unknown date")
    test_name = lab.get("test_name", "Unknown test")
    result = lab.get("result", "No result")
    ref_range = lab.get("reference_range", "N/A")
    interpretation = lab.get("interpretation", "N/A")

    narrative = (
        f"On {date}, lab results for {patient_name} showed {test_name}: {result}. "
        f"The reference range is {ref_range}, and the interpretation is {interpretation}."
    )
    return narrative


def doctor_note_to_narrative(note, patient_name):
    """Directly uses content with prepended metadata."""
    date = note.get("date", "Unknown date")
    author = note.get("author", "Unknown Author")
    content = note.get("content", "")
    return f"DATE: {date} | AUTHOR: {author} | CONTENT: {content}"


def pharmacy_note_to_narrative(note, patient_name):
    """Standard narrative for pharmacy records."""
    date = note.get("date", "Unknown date")
    pharmacy = note.get("pharmacy", "the pharmacy")
    content = note.get("content", "")
    return f"Pharmacy Record ({date}) at {pharmacy} for {patient_name}: {content}"


def clean_metadata(metadata):
    """Removes None or empty string values from metadata."""
    return {k: v for k, v in metadata.items() if v is not None and v != ""}


def patient_to_chunks(patient_data):
    """Splits patient data into atomic units for vector indexing."""
    patient_id = patient_data.get("patient_id")
    patient_name = patient_data.get("demographics", {}).get("name", "Unknown")
    chunks = []

    # Process visits
    for visit in patient_data.get("recent_visits", []):
        text = visit_to_narrative(visit, patient_name)
        metadata = clean_metadata(
            {
                "patient_id": patient_id,
                "timestamp": visit.get("date"),
                "event_type": "visit",
                "doctor": visit.get("doctor"),
            }
        )
        # We use visit_id for the deterministic UUID generation, but we don't store it in payload
        chunks.append(
            {"text": text, "metadata": metadata, "internal_id": visit.get("visit_id")}
        )

    # Process labs
    for lab in patient_data.get("lab_results", []):
        text = lab_to_narrative(lab, patient_name)
        metadata = clean_metadata(
            {
                "patient_id": patient_id,
                "timestamp": lab.get("date"),
                "event_type": "lab",
                "test_name": lab.get("test_name"),
            }
        )
        chunks.append(
            {"text": text, "metadata": metadata, "internal_id": lab.get("lab_id")}
        )

    # Process doctor_notes
    for note in patient_data.get("doctor_notes", []):
        text = doctor_note_to_narrative(note, patient_name)
        metadata = clean_metadata(
            {
                "patient_id": patient_id,
                "timestamp": note.get("date"),
                "event_type": "doctor_note",
                "author": note.get("author"),
            }
        )
        chunks.append(
            {"text": text, "metadata": metadata, "internal_id": note.get("note_id")}
        )

    # Process pharmacy_notes
    for note in patient_data.get("pharmacy_notes", []):
        text = pharmacy_note_to_narrative(note, patient_name)
        metadata = clean_metadata(
            {
                "patient_id": patient_id,
                "timestamp": note.get("date"),
                "event_type": "pharmacy_note",
                "pharmacy": note.get("pharmacy"),
            }
        )
        chunks.append(
            {"text": text, "metadata": metadata, "internal_id": note.get("entry_id")}
        )

    return chunks
