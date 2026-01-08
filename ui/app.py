import streamlit as st
import os
import json
import sys

# Add project root to sys.path to resolve core and utils modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vector_store import MedicalVectorStore
from core.agent import ClinicalAssistant
from langchain_core.messages import HumanMessage, AIMessage

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
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                # If message has tool calls, show them
                if (
                    hasattr(message, "additional_kwargs")
                    and "tool_calls" in message.additional_kwargs
                ):
                    tool_calls = message.additional_kwargs["tool_calls"]
                    if tool_calls:
                        for tc in tool_calls:
                            with st.expander(
                                f"🛠️ Tool Call: {tc['name']}", expanded=False
                            ):
                                st.write("**Input:**")
                                st.json(tc["args"])
                                if tc.get("response"):
                                    st.write("**Observation:**")
                                    st.text(tc["response"])
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                # If message has intermediate steps, show them
                if (
                    hasattr(message, "additional_kwargs")
                    and "intermediate_steps" in message.additional_kwargs
                ):
                    intermediate_steps = message.additional_kwargs["intermediate_steps"]
                    if intermediate_steps:
                        for action, observation in intermediate_steps:
                            with st.expander(
                                f"🛠️ Tool Call: {action.tool}", expanded=False
                            ):
                                st.write("**Input:**")
                                st.json(action.tool_input)
                                st.write("**Observation:**")
                                st.text(observation)
                    else:
                        st.caption("ℹ️ No tool calls needed - answered from context")
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                # If message has intermediate steps, show them
                if (
                    hasattr(message, "additional_kwargs")
                    and "intermediate_steps" in message.additional_kwargs
                ):
                    for action, observation in message.additional_kwargs[
                        "intermediate_steps"
                    ]:
                        with st.expander(f"🛠️ Tool Call: {action.tool}", expanded=False):
                            st.write("**Input:**")
                            st.code(action.tool_input)
                            st.write("**Observation:**")
                            st.markdown(observation)
                st.markdown(message.content)

    if prompt := st.chat_input("Ask about patient history, lab trends, etc."):
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare Identity Context for System Prompt
        identity_context = json.dumps(
            {"demographics": dem, "medical_history": med}, indent=2
        )

        executor = assistant.get_executor(identity_context, patient_id)

        with st.chat_message("assistant"):
            # Create placeholder for streaming
            message_placeholder = st.empty()
            tool_calls_placeholder = st.container()

            full_response = ""
            tool_calls_found = []

            # Stream the response
            with st.spinner("Analyzing records..."):
                # Use a unique thread_id per patient to maintain conversation context
                thread_id = f"patient_{patient_id}"

                for chunk in executor.stream(
                    {"messages": [("user", prompt)]},
                    config={"configurable": {"thread_id": thread_id}},
                    stream_mode="values",
                ):
                    messages = chunk.get("messages", [])

                    if messages:
                        last_msg = messages[-1]

                        # Check for tool calls
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                # Find corresponding response
                                tool_response = None
                                msg_index = messages.index(last_msg)
                                if msg_index + 1 < len(messages):
                                    next_msg = messages[msg_index + 1]
                                    if hasattr(next_msg, "content"):
                                        tool_response = next_msg.content

                                tool_call_data = {
                                    "name": tc.get("name"),
                                    "args": tc.get("args"),
                                    "response": tool_response,
                                }

                                if tool_call_data not in tool_calls_found:
                                    tool_calls_found.append(tool_call_data)

                        # Stream final AI response
                        if (
                            hasattr(last_msg, "content")
                            and type(last_msg).__name__ == "AIMessage"
                        ):
                            if (
                                not last_msg.tool_calls
                            ):  # Only show final answer, not tool call messages
                                full_response = last_msg.content
                                message_placeholder.markdown(full_response + "▌")

            # Show final response without cursor
            message_placeholder.markdown(full_response)

            # Show tool calls
            with tool_calls_placeholder:
                for tc in tool_calls_found:
                    with st.expander(f"🛠️ Tool Call: {tc['name']}", expanded=False):
                        st.write("**Input:**")
                        st.json(tc["args"])
                        if tc["response"]:
                            st.write("**Observation:**")
                            st.text(tc["response"])

            # Store AI message
            ai_msg = AIMessage(
                content=full_response,
                additional_kwargs={"tool_calls": tool_calls_found},
            )
            st.session_state.chat_history.append(ai_msg)
else:
    st.info("Please select a patient from the sidebar to begin.")
