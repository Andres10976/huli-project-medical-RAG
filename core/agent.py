from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from core.vector_store import MedicalVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.messages import SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()


class MedicalSearchSchema(BaseModel):
    query: str = Field(description="Semantic search query for clinical history or labs")
    patient_id: str = Field(description="The unique ID of the patient")
    event_type: Optional[str] = Field(None, description="Filter by 'visit' or 'lab'")


class MedicalSearchTool(BaseTool):
    name: str = "medical_search_tool"
    description: str = (
        "Search for specific clinical events, visit notes, or lab trends for a patient."
    )
    args_schema: Type[BaseModel] = MedicalSearchSchema
    vector_store: MedicalVectorStore = None

    def __init__(self, vector_store):
        super().__init__()
        self.vector_store = vector_store

    def _run(self, query: str, patient_id: str, event_type: Optional[str] = None):
        results = self.vector_store.search(query, patient_id, event_type=event_type)
        if not results:
            return "No relevant records found for this query."

        formatted_results = []
        for res in results:
            event_type = res.payload.get("event_type", "record")
            formatted_results.append(
                f"Source: {event_type.capitalize()}\nDate: {res.payload['timestamp']}\nContent: {res.payload['text']}"
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
        self.tool = MedicalSearchTool(vector_store=vector_store)
        self.tools = [self.tool]

    def get_executor(self, identity_context):
        # Escape braces in identity_context because ChatPromptTemplate will try to parse them
        escaped_context = identity_context.replace("{", "{{").replace("}", "}}")
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are a professional clinical assistant. 
Current Patient Identity Context:
{escaped_context}

INSTRUCTIONS:
1. Use the `medical_search_tool` to look up specific clinical history or lab trends. 
2. Always cite the date and source of the information in your answer (e.g., "Según visita del 2024-10-15...").
3. It must be absolutely clear where each piece of data comes from.
4. If the patient has allergies (shown in the identity context), always be mindful of them when discussing treatments.""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent, tools=self.tools, verbose=True, return_intermediate_steps=True
        )
