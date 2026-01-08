# Clinical Assistant RAG (EHR-Driven)

A production-ready medical assistant powered by Retrieval-Augmented Generation (RAG) for querying Electronic Health Records (EHR). Built with semantic search, LLM-based reasoning, and real-time patient data retrieval.

## 🎯 Overview

This system enables healthcare professionals to query patient medical records through natural language, combining:
- **Semantic Search**: Retrieve relevant clinical events from patient history
- **Context-Aware Responses**: Static identity information (demographics, allergies, medications) always available
- **Medical-Grade Embeddings**: Specialized embeddings optimized for clinical text
- **Tool-Based Architecture**: LangChain agent with structured search capabilities

## ✨ Key Features

### 1. **Identity vs. Event Data Split**
- **Static Context** (Always in Context Window): Demographics, chronic conditions, allergies, current medications
- **Dynamic Events** (Retrieved via Search): Visits, lab results, doctor notes, clinical observations
- **Rationale**: Separates critical static info from voluminous event data, optimizing context window usage

### 2. **Atomic Event Chunking**
- Each visit and lab result is indexed as a separate vector
- Structured narrative format for optimal semantic matching
- Enables precise retrieval of specific clinical events

### 3. **Smart Search Capabilities**
- Semantic similarity search using medical-grade embeddings
- Event type filtering (`visit` vs `lab`)
- Chronological ordering for temporal queries ("most recent", "latest")
- Patient-scoped search with metadata filtering

### 4. **Clinical-Grade LLM**
- DeepSeek-Chat via OpenAI-compatible API
- Temperature set to 0 for consistent, factual responses
- Strict hallucination prevention instructions
- Source citation for all retrieved information

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Streamlit UI                             │
│  (Patient Selector | Identity Banner | Chat Interface)          │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ├─────────────────────────────────────────┐
                        │                                         │
                        ▼                                         ▼
┌───────────────────────────────┐                   ┌───────────────────────────────┐
│   Patient JSON Files         │                   │   Clinical Assistant Agent   │
│   (data/*.json)              │                   │   (LangChain + DeepSeek)      │
│   - demographics             │                   │                               │
│   - medical_history          │◄──────────────────│   System Prompt + Tools       │
│   - recent_visits            │                   │   - medical_search_tool       │
│   - lab_results              │                   │                               │
└───────────────────────────────┘                   └───────────────┬───────────────┘
                        │                                         │
                        ▼                                         ▼
┌───────────────────────────────┐                   ┌───────────────────────────────┐
│   Ingestion Pipeline          │                   │   MedicalVectorStore         │
│   (scripts/ingest_data.py)    │                   │   (Qdrant + Voyage AI)       │
│   - JSON parsing              │                   │                               │
│   - Narrative transformation  │───┐               │   - 512-dim embeddings       │
│   - Chunk generation          │   │               │   - Cosine similarity        │
│   - Vector embeddings         │   │               │   - Patient filtering        │
└───────────────────────────────┘   │               │   - Event type filtering     │
                                     │               │   - Date ordering             │
                                     │               └───────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Vector Database (Qdrant)                     │
│  Collection: medical_records                                     │
│  - Vector dimension: 512                                         │
│  - Distance: Cosine                                             │
│  - Payload indexes: patient_id, event_type, timestamp          │
│  - Deterministic UUIDs (UUID v5)                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Technical Decisions

### 1. **Voyage AI (voyage-3.5) for Embeddings**
**Decision**: Use Voyage AI's `voyage-3.5` model with 512 dimensions for medical embeddings

**Rationale**:
- Specialized for medical/clinical text vs. general-purpose embeddings
- Optimized for healthcare terminology and medical concepts
- 512 dimensions provides good balance between performance and storage
- Outperforms general models on medical semantic search tasks

**Alternatives Considered**:
- OpenAI `text-embedding-3-small` (512 dim): General-purpose, less medical-specific
- OpenAI `text-embedding-3-large` (3072 dim): Overkill, expensive
- Cohere `embed-english-v3.0`: Good but not medical-specialized

### 2. **DeepSeek-Chat via OpenAI-Compatible API**
**Decision**: Use DeepSeek-Chat as the LLM, accessed through OpenAI-compatible API interface

**Rationale**:
- Cost-effective compared to GPT-4/Claude Opus
- Strong reasoning capabilities for clinical tasks
- OpenAI-compatible API enables easy integration with LangChain
- Temperature=0 ensures consistent, factual responses critical for medical use

**Configuration**:
```python
ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0,  # Critical for medical accuracy
)
```

### 3. **Qdrant for Vector Database**
**Decision**: Use Qdrant as the vector database backend

**Rationale**:
- Native Python client with clean API
- Excellent filtering capabilities with payload indexes
- Efficient similarity search with cosine distance
- Docker-based deployment for easy setup
- In-memory persistence via volume mounting
- Supports deterministic UUIDs for point deduplication

**Key Features Used**:
- Payload indexes on `patient_id`, `event_type`, `timestamp`
- Cosine distance for semantic similarity
- Deterministic UUID v5 for reproducible point IDs

### 4. **Identity vs. Event Data Separation**
**Decision**: Split patient data into static (identity) and dynamic (events) categories

**Rationale**:
- **Identity Context** (demographics, allergies, conditions, medications): Always present in system prompt for immediate reference
- **Event Data** (visits, labs): Retrieved on-demand via semantic search
- Reduces context window usage
- Ensures critical safety information (allergies) is always visible
- Enables efficient retrieval of relevant historical events

**Implementation**:
```python
identity_context = {
    "demographics": {...},  # Always in context
    "medical_history": {
        "chronic_conditions": [...],
        "allergies": [...],  # Critical safety info
        "current_medications": [...]
    }
}

# Events retrieved via tool
events = medical_search_tool(query="diabetes management", event_type="visit")
```

### 5. **Atomic Event Chunking**
**Decision**: Index each visit and lab result as a separate vector chunk

**Rationale**:
- Enables precise retrieval of specific clinical events
- Better semantic matching at event granularity
- Allows filtering by event type (visit vs lab)
- Facilitates chronological ordering for temporal queries

**Narrative Format**:
```python
# Visit: DATE | DOCTOR | REASON | NOTES
"DATE: 2024-10-15 | DOCTOR: Dra. Martínez | REASON: Control rutinario | NOTES: Glucosa en ayunas: 128 mg/dL..."

# Lab: DATE | TEST | RESULTS
"DATE: 2024-10-10 | TEST: Panel metabólico | glucose: 128 mg/dL | hba1c: 7.2%..."
```

### 6. **Tool-Based Agent Architecture**
**Decision**: Implement LangChain agent with `medical_search_tool` for all historical queries

**Rationale**:
- Enforces disciplined retrieval - agent cannot hallucinate event data
- Clear separation of capabilities (static context vs. search)
- Enables transparent tool calls for audit trail
- Structured interface with validation (event_type must be 'visit' or 'lab')
- Date ordering option for temporal queries

**Tool Schema**:
```python
class MedicalSearchSchema(BaseModel):
    query: str  # Semantic search query
    event_type: str  # REQUIRED: 'visit' or 'lab'
    order_by_date: bool  # True for "most recent" queries
```

### 7. **Deterministic UUID v5 for Point IDs**
**Decision**: Use UUID v5 with namespace-based generation for vector point IDs

**Rationale**:
- Reproducible point IDs from same input data
- Enables idempotent upserts (same data = same ID)
- Prevents duplicate entries on re-ingestion
- Namespace-based collision avoidance across patients

**Implementation**:
```python
NAMESPACE_MEDICAL = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
point_id = uuid.uuid5(NAMESPACE_MEDICAL, f"{patient_id}_{internal_id}")
```

### 8. **Payload Indexes for Efficient Filtering**
**Decision**: Create indexes on `patient_id`, `event_type`, and `timestamp` fields

**Rationale**:
- Enables efficient patient-scoped searches
- Supports event type filtering without full scans
- Allows chronological sorting without post-processing
- Critical for multi-patient system scalability

### 9. **Thread-Based Conversation Context**
**Decision**: Use `thread_id` parameter to maintain conversation history per patient

**Rationale**:
- Preserves context across multiple questions about same patient
- Enables follow-up questions
- Separate threads for different patients
- Better user experience in clinical workflow

**Implementation**:
```python
thread_id = f"patient_{patient_id}"
executor.stream(
    {"messages": [("user", prompt)]},
    config={"configurable": {"thread_id": thread_id}},
    stream_mode="values"
)
```

### 10. **Streamlit for UI**
**Decision**: Build web interface with Streamlit

**Rationale**:
- Fast development time for interview project
- Built-in chat interface components
- Easy sidebar for patient selection
- Session state management for chat history
- Tool call visualization for transparency
- No frontend build pipeline needed

### 11. **Structured System Prompt**
**Decision**: Comprehensive system prompt with explicit instructions and constraints

**Rationale**:
- Enforces strict behavior: always search, never hallucinate
- Includes current datetime for relative time queries
- Clear separation of identity vs. event data
- Explicit error handling instructions
- Citation requirements for source attribution

**Key Instructions**:
- "MUST use medical_search_tool for ANY question about past visits, labs..."
- "NEVER make up, invent, or hallucinate medical data"
- "If you cannot find the requested information, say: No encontré esa información..."
- "Always cite the exact date and source of information"

### 12. **Docker Compose for Qdrant**
**Decision**: Deploy Qdrant via Docker Compose with volume persistence

**Rationale**:
- Simple, reproducible deployment
- Volume mounting ensures data persistence across container restarts
- Standardized environment for development and production
- Easy to scale (add replicas if needed)

## 📊 Data Flow

### Query Execution Flow
```
1. User submits question via Streamlit UI
   ↓
2. UI loads patient identity context (demographics, medical_history)
   ↓
3. ClinicalAssistant creates agent with system prompt + identity context
   ↓
4. Agent processes question and calls medical_search_tool
   ↓
5. MedicalSearchTool queries MedicalVectorStore
   ↓
6. MedicalVectorStore:
   a. Embeds query using Voyage AI (512 dim)
   b. Filters by patient_id (required)
   c. Optionally filters by event_type ('visit' or 'lab')
   d. Performs cosine similarity search in Qdrant
   e. Optionally sorts by timestamp (if order_by_date=True)
   ↓
7. Returns formatted results to agent
   ↓
8. Agent synthesizes response using:
   - Identity context (always available)
   - Retrieved event data (from search results)
   - Explicit citations to sources
   ↓
9. Streamlit displays response with tool call details
```

### Data Ingestion Flow
```
1. User runs python scripts/ingest_data.py
   ↓
2. Script reads all JSON files from data/ directory
   ↓
3. For each patient:
   a. Parse JSON structure
   b. Transform visits to narrative format (DATE | DOCTOR | REASON | NOTES)
   c. Transform labs to narrative format (DATE | TEST | RESULTS)
   d. Generate metadata (patient_id, timestamp, event_type, etc.)
   e. Create chunks with internal_id (visit_id, lab_id)
   ↓
4. MedicalVectorStore:
   a. Generates embeddings via Voyage AI (512 dim)
   b. Creates deterministic UUID v5 point IDs
   c. Upserts points to Qdrant collection
   ↓
5. Indexes created on patient_id, event_type, timestamp
   ↓
6. Vector DB ready for queries
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- API keys for DeepSeek and Voyage AI

### Installation

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd huli-project-medical-RAG
```

#### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Configure Environment Variables
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
VOYAGE_API_KEY=your_voyage_api_key_here
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
```

#### 5. Start Qdrant (Vector Database)
```bash
docker-compose up -d
```

Verify Qdrant is running:
```bash
curl http://localhost:6333/collections
```

#### 6. Initial Data Ingestion (First Run Only)
```bash
python scripts/ingest_data.py
```

Expected output:
```
🚀 Starting First Run Ingestion...
Checking collection 'medical_records'...
Collection exists. Recreating it to ensure 512-dimension configuration...
📄 Processing patient_1.json...
✅ Ingested 3 chunks for Juan Pérez
📄 Processing patient_2.json...
✅ Ingested 4 chunks for María González

✨ First run ingestion complete! Your Vector DB is ready.
```

#### 7. Run the Application
```bash
streamlit run ui/app.py
```

The application will open at `http://localhost:8501`

## 💻 Usage

### Patient Selection
1. Select a patient from the sidebar dropdown
2. Identity banner displays:
   - Patient name, age, gender
   - Chronic conditions
   - Allergies (highlighted in red if present)

### Query Examples

**General Questions**:
- "What is the patient's latest lab results?"
- "Show me the most recent visit"
- "What medications is the patient currently taking?"
- "Does the patient have any allergies?"

**Temporal Queries**:
- "What happened in the last visit?"
- "Show me labs from 2 months ago"
- "What's the trend in glucose levels?"

**Clinical Questions**:
- "How is the diabetes being managed?"
- "What treatments have been prescribed?"
- "Any concerns mentioned in recent visits?"

### Tool Call Visualization
- Expand "🛠️ Tool Call" sections to see:
  - Search query used
  - Event type filter applied
  - Retrieved results (timestamps and content)

### Switching Patients
- Changing patients clears chat history and loads new identity context
- Each patient has separate conversation thread

## 📁 Project Structure

```
huli-project-medical-RAG/
├── core/
│   ├── agent.py              # LangChain agent with medical search tool
│   └── vector_store.py       # Qdrant integration + Voyage embeddings
├── utils/
│   └── narrative.py          # Data transformation to narrative format
├── ui/
│   └── app.py                # Streamlit web interface
├── scripts/
│   └── ingest_data.py       # ETL pipeline for vector DB ingestion
├── data/
│   ├── patient_1.json        # Sample patient data
│   ├── patient_2.json        # Sample patient data
│   └── example.json          # Data format template
├── qdrant_storage/           # Persisted vector data (Docker volume)
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
├── docker-compose.yml        # Qdrant container configuration
└── README.md                 # This file
```

## 🔍 Key Components

### MedicalVectorStore (`core/vector_store.py`)
- Manages Qdrant client connection
- Generates embeddings via Voyage AI
- Creates and configures collections with payload indexes
- Provides search with patient filtering and event type filtering

### ClinicalAssistant (`core/agent.py`)
- Creates LangChain agent with DeepSeek LLM
- Implements `medical_search_tool` for historical queries
- Constructs system prompts with identity context
- Manages agent execution with thread-based context

### Narrative Utils (`utils/narrative.py`)
- Transforms structured JSON to narrative format
- Converts visits: `DATE | DOCTOR | REASON | NOTES`
- Converts labs: `DATE | TEST | RESULTS`
- Generates metadata for vector payloads

### Streamlit UI (`ui/app.py`)
- Patient selection sidebar
- Identity context banner
- Chat interface with message history
- Tool call visualization
- Real-time streaming responses

### Ingestion Script (`scripts/ingest_data.py`)
- Processes all JSON files in `data/` directory
- Transforms data to narrative format
- Generates embeddings and upserts to Qdrant
- Handles collection recreation for dimension updates

## 🧪 Testing

### Manual Testing Workflow

1. **Test Vector Store**:
```python
from core.vector_store import MedicalVectorStore

vs = MedicalVectorStore()
results = vs.search("diabetes", patient_id="P001", event_type="visit", limit=3)
print(results)
```

2. **Test Agent**:
```python
from core.agent import ClinicalAssistant
import json

vs = MedicalVectorStore()
assistant = ClinicalAssistant(vs)

identity_context = json.dumps({
    "demographics": {...},
    "medical_history": {...}
})

executor = assistant.get_executor(identity_context, patient_id="P001")
response = executor.invoke({"messages": [("user", "What medications?")]})
print(response)
```

3. **Test Ingestion**:
```bash
python scripts/ingest_data.py
```

### Test Cases to Verify

- [ ] Patient selection switches correctly
- [ ] Identity context displays properly
- [ ] Chat history clears on patient change
- [ ] Search returns relevant results
- [ ] Event type filtering works
- [ ] Date ordering works for temporal queries
- [ ] Tool calls are displayed correctly
- [ ] Allergy warnings show in red
- [ ] LLM cites sources correctly
- [ ] No hallucinations for missing data

## 🔧 Configuration

### Voyage AI Model
- **Model**: `voyage-3.5`
- **Dimensions**: 512
- **Rationale**: Medical-specialized embeddings, good performance/size balance

### DeepSeek LLM
- **Model**: `deepseek-chat`
- **Temperature**: 0 (factual, consistent)
- **API Base**: `https://api.deepseek.com/v1`

### Qdrant Collection
- **Name**: `medical_records`
- **Vector Size**: 512
- **Distance**: Cosine
- **Indexes**: `patient_id`, `event_type`, `timestamp`

### Chunk Size
- **Strategy**: One chunk per event (visit or lab)
- **Rationale**: Atomic events enable precise retrieval
- **No overlapping or recursive chunking**

## 🐛 Troubleshooting

### Qdrant Connection Issues
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Check logs
docker logs medical-rag-qdrant

# Restart Qdrant
docker-compose restart
```

### API Key Errors
- Verify `DEEPSEEK_API_KEY` in `.env`
- Verify `VOYAGE_API_KEY` in `.env`
- Check API key validity and credits

### No Search Results
- Verify data ingestion completed successfully
- Check patient_id matches exactly
- Ensure collection exists: `curl http://localhost:6333/collections`
- Try different query terms

### Import Errors
```bash
# Ensure you're in project root directory
cd huli-project-medical-RAG

# Activate virtual environment
source .venv/bin/activate

# Verify installation
pip list | grep -E "qdrant|voyage|langchain|streamlit"
```

## 🚧 Future Enhancements

### Potential Improvements
1. **Multi-language Support**: Add bilingual prompts (English/Spanish)
2. **Advanced Analytics**: Trend analysis for lab values over time
3. **Multi-patient Queries**: Compare patients with similar conditions
4. **Document Upload**: Support for PDF/CSV medical reports
5. **Real-time Data Sync**: Integration with EHR systems
6. **User Authentication**: Role-based access control
7. **Audit Logging**: Track all queries and tool calls
8. **Explainability**: Highlight retrieved text segments in UI

### Performance Optimizations
1. **Caching**: Cache frequent queries
2. **Batch Embeddings**: Process multiple chunks in parallel
3. **Query Rewriting**: Optimize queries for better retrieval
4. **Hybrid Search**: Combine semantic with keyword search
5. **Reranking**: Add reranking layer for improved relevance

## 📝 Data Format

### Patient JSON Structure
```json
{
  "patient_id": "P001",
  "demographics": {
    "name": "Juan Pérez",
    "age": 45,
    "gender": "M",
    "blood_type": "O+"
  },
  "medical_history": {
    "chronic_conditions": ["Diabetes Tipo 2", "Hipertensión"],
    "allergies": ["Penicilina"],
    "current_medications": [
      {
        "name": "Metformina",
        "dose": "850mg",
        "frequency": "2x/día"
      }
    ]
  },
  "recent_visits": [
    {
      "date": "2024-10-15",
      "reason": "Control rutinario",
      "notes": "Glucosa en ayunas: 128 mg/dL...",
      "doctor": "Dra. Martínez",
      "visit_id": "V001"
    }
  ],
  "lab_results": [
    {
      "date": "2024-10-10",
      "test": "Panel metabólico",
      "results": {
        "glucose": "128 mg/dL",
        "hba1c": "7.2%"
      },
      "lab_id": "L001"
    }
  ]
}
```

## 🤝 Contributing

This is an interview project demonstrating RAG architecture for medical applications.

## 📄 License

[Specify your license here]

## 🙏 Acknowledgments

- **Voyage AI**: Medical-grade embeddings
- **DeepSeek**: Cost-effective LLM with strong reasoning
- **Qdrant**: High-performance vector database
- **LangChain**: Agent framework and tools
- **Streamlit**: Rapid UI development

---

**Built with ❤️ for better clinical decision support**
