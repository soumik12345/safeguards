import importlib

import streamlit as st
import weave

from guardrails_genie.utils import initialize_guardrails_on_playground


def initialize_session_state():
    if "guardrails" not in st.session_state:
        st.session_state.guardrails = []
    if "guardrail_names" not in st.session_state:
        st.session_state.guardrail_names = []
    if "guardrails_manager" not in st.session_state:
        st.session_state.guardrails_manager = None
    if "initialize_guardrails_button" not in st.session_state:
        st.session_state.initialize_guardrails_button = False
    if "start_chat_button" not in st.session_state:
        st.session_state.start_chat_button = False
    if "prompt" not in st.session_state:
        st.session_state.prompt = ""
    if "test_guardrails_button" not in st.session_state:
        st.session_state.test_guardrails_button = False

    if "prompt_injection_llm_model" not in st.session_state:
        st.session_state.prompt_injection_llm_model = None
    if "prompt_injection_llama_guard_checkpoint_name" not in st.session_state:
        st.session_state.prompt_injection_llama_guard_checkpoint_name = None


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

initialize_guardrails_button = st.sidebar.button("Initialize Guardrails")
st.session_state.initialize_guardrails_button = (
    initialize_guardrails_button
    if not st.session_state.initialize_guardrails_button
    else st.session_state.initialize_guardrails_button
)

if st.session_state.initialize_guardrails_button:
    initialize_guardrails_on_playground()

    prompt = st.text_area("User Prompt", value="")
    st.session_state.prompt = prompt

    test_guardrails_button = st.button("Test Guardrails")
    st.session_state.test_guardrails_button = test_guardrails_button

    if st.session_state.test_guardrails_button:
        st.write("Maza Ayega")
