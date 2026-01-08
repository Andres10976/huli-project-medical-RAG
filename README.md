# Clinical Assistant RAG (EHR-Driven)

This project implements a real-time medical assistant using RAG (Retrieval-Augmented Generation) based on patient EHR (Electronic Health Record) data.

## Features

- **Identity vs. Event Split**: Static demographics and allergies are always in the context window.
- **Atomic Event Chunking**: Visits and labs are converted to narratives and indexed in Qdrant.
- **DeepSeek Integration**: Uses DeepSeek-Chat for clinical reasoning.
- **Medical Embeddings**: Powered by `voyage-3.5`.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Qdrant**:
   ```bash
   docker-compose up -d
   ```

3. **Environment Variables**:
   Create a `.env` file from the template and add your API keys:
   - `DEEPSEEK_API_KEY`
   - `VOYAGE_API_KEY`

4. **Initial Ingestion (First Run)**:
   Process all existing JSON files and ingest them into Qdrant:
   ```bash
   python3 scripts/ingest_data.py
   ```

5. **Run the App**:
   ```bash
   streamlit run ui/app.py
   ```

## Project Structure

- `data/`: JSON patient records.
- `core/`: Vector store and Agent logic.
- `utils/`: Narrative transformation logic.
- `ui/`: Streamlit dashboard.
