import streamlit as st
import os
import json
from core.vector_store import MedicalVectorStore
from core.agent import ClinicalAssistant
from langchain.schema import HumanMessage, AIMessage

st.set_page_config(page_title="Clinical Assistant RAG", layout="wide")


# Initialize Vector Store and Assistant
@st.cache_resource
def get_resources():
    vs = MedicalVectorStore()
    assistant = ClinicalAssistant(vs)
    return vs, assistant


vector_store, assistant = get_resources()

# Sidebar: Patient Selection
st.sidebar.title("Patient Selector")


def get_patient_list():
    patients = []
    data_dir = "data"
    for filename in os.listdir(data_dir):
        if filename.endswith(".json") and filename != "example.json":
            with open(os.path.join(data_dir, filename), "r") as f:
                try:
                    data = json.load(f)
                    name = data.get("demographics", {}).get("name", "Unknown")
                    p_id = data.get("patient_id", "No ID")
                    patients.append(
                        {
                            "display": f"{name} - {p_id}",
                            "filename": filename,
                            "id": p_id,
                        }
                    )
                except:
                    continue
    return sorted(patients, key=lambda x: x["display"])


patient_options = get_patient_list()
selected_patient = st.sidebar.selectbox(
    "Select Patient", patient_options, format_func=lambda x: x["display"]
)

st.sidebar.divider()
st.sidebar.subheader("System Status")
st.sidebar.success("CDC Pipeline: Active & Watching")
st.sidebar.caption("Vector DB: Qdrant Connected")

if selected_patient:
    filename = selected_patient["filename"]
    with open(os.path.join("data", filename), "r") as f:
        patient_data = json.load(f)

    patient_id = patient_data["patient_id"]

    # Check if patient changed to clear history
    if (
        "current_patient_id" not in st.session_state
        or st.session_state.current_patient_id != patient_id
    ):
        st.session_state.current_patient_id = patient_id
        st.session_state.chat_history = []

    # Identity Banner (Static Context)
    dem = patient_data["demographics"]
    med = patient_data["medical_history"]

    st.markdown(f"## Patient: {dem['name']} ({dem['age']}y, {dem['gender']})")

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Chronic Conditions:** {', '.join(med['chronic_conditions'])}")
    with col2:
        if med["allergies"]:
            st.error(f"**⚠️ ALLERGIES:** {', '.join(med['allergies'])}")
        else:
            st.success("**Allergies:** None reported")

    st.divider()

    # Chat Interface
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(message.content)

    if prompt := st.chat_input("Ask about patient history, lab trends, etc."):
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare Identity Context for System Prompt
        identity_context = json.dumps(
            {"demographics": dem, "medical_history": med}, indent=2
        )

        executor = assistant.get_executor(identity_context)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing records..."):
                response = executor.invoke(
                    {
                        "input": prompt,
                        "chat_history": st.session_state.chat_history[:-1],
                    }
                )
                output = response["output"]
                st.markdown(output)
                st.session_state.chat_history.append(AIMessage(content=output))
else:
    st.info("Please select a patient from the sidebar to begin.")
