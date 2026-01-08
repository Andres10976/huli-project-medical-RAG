# Project Roadmap: Clinical Assistant RAG (EHR-Driven)

## 1\. Data Strategy: "Identity vs. Event" Split

To handle large patient files without blowing the context window, we split the JSON data into two distinct layers:

### A. The "Golden" Identity Context (Static)

This is the "Always-On" data. It is never searched via vector DB; it is directly injected into the LLM System Prompt.

- **Fields:** `demographics`, `medical_history.chronic_conditions`, `medical_history.allergies`.
- **Reasoning:** A doctor should never have to "search" if a patient is allergic to Penicillin. The LLM must know this from the first token.
- **Implementation:** When a patient is selected in the UI, this data is fetched by ID and placed in the `SystemMessage`.

### B. Atomic Event Chunking (Dynamic/Vector)

We treat every visit and lab result as an **Atomic Narrative Unit**.

- **Chunking Logic:** - 1 `recent_visit` = 1 Vector Document.
  - 1 `lab_result` = 1 Vector Document.

- **Narrative Transformation:** To avoid "table trash," we convert the JSON into a natural language string before embedding.
  - _Input:_ `{ "date": "2024-10-15", "notes": "PA: 135/85", ... }`
  - _Narrative:_ "On 2024-10-15, the patient had a routine checkup with Dra. Martínez. Clinical notes: PA: 135/85. Losartan dosage was adjusted."

## 2\. Vector DB & Embeddings (Qdrant + Voyage)

- **Provider:** Self-hosted Qdrant (Docker).
- **Embedding Model:** `voyage-medical-2` (via Voyage AI API).
- **Metadata Schema:**
  - `patient_id` (UUID): For strict filtering.
  - `timestamp` (Unix): For "Sort by date" queries.
  - `event_type` (visit | lab): For targeted search tools.
  - `source_json`: The raw snippet for 100% accurate grounding.

## 3\. The CDC (Change Data Capture) Pipeline

Since we are mocking a real-time EHR, we will implement a file-watcher:

1.  **Watchdog:** A Python background process monitors `data/*.json`.
2.  **Hash Check:** The system maintains a local state of `last_indexed_hash`.
3.  **Upsert Logic:** If a file changes, the system identifies the specific `visit_id` or `lab_id` that is new or updated and performs a `point_upsert` in Qdrant.
4.  **Real-time Feedback:** The UI reflects "Indexing updated" when the doctor saves a new note in the JSON.

## 4\. LangChain Tooling Strategy

We will build a single `MedicalSearchTool` with the following parameters:

- `query`: The semantic search string.
- `patient_id`: Mandatory filter.
- `start_date` / `end_date`: Optional range filters (converted to Qdrant filters).

**Prompt Logic:**

> "You are a clinical assistant. You are currently seeing **{IdentityContext}**. Use the `MedicalSearchTool` to look up specific history or lab trends. **Always** cite the date of the event in your answer."

## 5\. UI/UX Recommendation

For a production-ready feel for medics:

- **Framework:** Next.js + Tailwind + shadcn/ui.
- **Template:** A "Dashboard" style layout.
  - **Left Sidebar:** Patient Selector (Card list with Names/IDs).
  - **Top Bar:** Current Patient Banner (Showing name, age, and **Red Flashing Alert** for Allergies).
  - **Main Area:** Chat interface with "Source Cards" appearing below LLM responses.

- **Reflex:** The chat should automatically reset its memory when the `patient_id` changes in the selector.
