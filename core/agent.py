from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field
from core.vector_store import MedicalVectorStore
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.schema import SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()


class MedicalSearchSchema(BaseModel):
    query: str = Field(
        description="Semantic search query for clinical history, labs, doctor notes, or pharmacy records"
    )
    patient_id: str = Field(description="The unique ID of the patient")
    event_type: Optional[str] = Field(
        None, description="Filter by 'visit', 'lab', 'doctor_note', or 'pharmacy_note'"
    )


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
            formatted_results.append(
                f"Date: {res.payload['timestamp']}\nContent: {res.payload['text']}"
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
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are a professional clinical assistant. 
Current Patient Identity Context:
{identity_context}

Use the `medical_search_tool` to look up specific clinical history or lab trends. 
Always cite the date of the event in your answer. 
If the patient has allergies (shown in the identity context), always be mindful of them when discussing treatments.""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
