import importlib

import streamlit as st
import weave
from guardrails_genie.guardrails import GuardrailManager
from guardrails_genie.llm import OpenAIModel


def initialize_session_state():
    if "guardrails" not in st.session_state:
        st.session_state.guardrails = []
    if "guardrail_names" not in st.session_state:
        st.session_state.guardrail_names = []


weave.init(project_name="guardrails_genie")
initialize_session_state()
st.title(":material/robot: Guardrails Genie Playground")

openai_model = st.sidebar.selectbox(
    "OpenAI LLM for Chat", ["", "gpt-4o-mini", "gpt-4o"]
)

guardrail_names = st.sidebar.multiselect(
    label="Select Guardrails",
    options=[
        cls_name
        for cls_name, cls_obj in vars(
            importlib.import_module("guardrails_genie.guardrails")
        ).items()
        if isinstance(cls_obj, type) and cls_name != "GuardrailManager"
    ],
)
st.session_state.guardrail_names = guardrail_names
