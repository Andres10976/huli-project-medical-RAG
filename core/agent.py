from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from core.vector_store import MedicalVectorStore
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class MedicalSearchSchema(BaseModel):
    query: str = Field(description="Semantic search query for clinical history or labs")
    event_type: str = Field(description="REQUIRED: Must be either 'visit' or 'lab'")
    order_by_date: bool = Field(
        default=False,
        description="Set to True for queries about 'most recent', 'latest', 'last' events to ensure chronological ordering",
    )


class MedicalSearchTool(BaseTool):
    name: str = "medical_search_tool"
    description: str = (
        "REQUIRED tool to search patient's visit notes, lab results, and clinical events. "
        "You MUST use this tool for ANY question about past visits, lab values, doctor notes, "
        "diagnoses, treatments, or historical medical data. "
        "DO NOT answer from memory - always search the actual records. "
        "IMPORTANT: You MUST specify event_type as either 'visit' or 'lab'. "
        "For queries about 'most recent', 'latest', 'last' events, set order_by_date=True."
    )
    args_schema: Type[BaseModel] = MedicalSearchSchema
    vector_store: MedicalVectorStore = None
    patient_id: str = None  # Will be set when creating the tool

    def __init__(self, vector_store, patient_id=None):
        super().__init__()
        self.vector_store = vector_store
        self.patient_id = patient_id

    def _run(self, query: str, event_type: str, order_by_date: bool = False):
        if not self.patient_id:
            return "Error: Patient ID not set"

        # Validate event_type
        if event_type not in ["visit", "lab"]:
            return "Error: event_type must be either 'visit' or 'lab'"

        results = self.vector_store.search(
            query, self.patient_id, event_type=event_type, order_by_date=order_by_date
        )
        if not results:
            return "No relevant records found for this query."

        formatted_results = []
        for res in results:
            event_type_str = res.payload.get("event_type", "record")
            formatted_results.append(
                f"Source: {event_type_str.capitalize()}\nDate: {res.payload['timestamp']}\nContent: {res.payload['text']}"
            )

        return "\n\n---\n\n".join(formatted_results)


class ClinicalAssistant:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        # Using DeepSeek via OpenAI-compatible API
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base="https://api.deepseek.com/v1",
            temperature=0,
        )

    def get_executor(self, identity_context, patient_id):
        # Create tool with patient_id bound to it
        tool = MedicalSearchTool(vector_store=self.vector_store, patient_id=patient_id)
        tools = [tool]

        # Get current datetime for relative time understanding
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # System prompt with instructions
        system_prompt = f"""You are a professional clinical assistant with access to a patient's medical records database.

CURRENT DATE AND TIME: {current_datetime}
(Use this for understanding relative time queries like "2 months ago", "last week", "recent", etc.)

Current Patient Identity Context (STATIC INFO ONLY):
{identity_context}

CRITICAL INSTRUCTIONS:
1. The identity context above contains ONLY basic demographics, chronic conditions, allergies, and current medications.
2. For ANY question about visit history, lab results, doctor notes, symptoms, diagnoses, or past events, you MUST use the medical_search_tool.
3. When using medical_search_tool, you MUST specify event_type as either 'visit' or 'lab'.
4. For queries about "most recent", "latest", "last" events, set order_by_date=True to ensure chronological ordering.
5. NEVER make up, invent, or hallucinate medical data. Only report information you have explicitly found.
6. If you cannot find the requested information after searching, you MUST say "No encontré esa información en los registros disponibles" - DO NOT guess or make up an answer.
7. Always cite the exact date and source of information (e.g., "Según visita del 2024-10-15...").
8. It must be absolutely clear where each piece of data comes from.
9. If the patient has allergies (shown in identity context), always be mindful when discussing treatments.
10. Ground ALL your answers in either the identity context or tool search results.

Remember: Visit notes and lab results are NOT in the identity context - you must search for them!"""

        agent = create_agent(
            self.llm, tools, system_prompt=system_prompt
        )

        return agent