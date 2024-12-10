import streamlit as st

import wandb


def initialize_session_state():
    if "weave_project_name" not in st.session_state:
        st.session_state.weave_project_name = "guardrails-genie"
    if "weave_entity_name" not in st.session_state:
        st.session_state.weave_entity_name = ""
    if "wandb_api_key" not in st.session_state:
        st.session_state.wandb_api_key = ""
    if "authenticate_button" not in st.session_state:
        st.session_state.authenticate_button = False
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False


initialize_session_state()
st.title("🧞‍♂️ Guardrails Genie")

st.write(
    "Guardrails-Genie is a tool that helps you implement guardrails in your LLM applications."
)

with st.expander("Login to Your WandB Account", expanded=True):
    st.markdown(
        "Get your Wandb API key from [https://wandb.ai/authorize](https://wandb.ai/authorize)"
    )
    weave_entity_name = st.text_input(
        "Weave Entity Name", value=st.session_state.weave_entity_name
    )
    st.session_state.weave_entity_name = weave_entity_name
    weave_project_name = st.text_input(
        "Weave Project Name", value=st.session_state.weave_project_name
    )
    st.session_state.weave_project_name = weave_project_name
    wandb_api_key = st.text_input("Wandb API Key", value="", type="password")
    st.session_state.wandb_api_key = wandb_api_key
    authenticate_button = st.button("Authenticate")
    st.session_state.authenticate_button = authenticate_button

    if authenticate_button and (
        st.session_state.wandb_api_key != ""
        and st.session_state.weave_project_name != ""
    ):
        is_wandb_logged_in = wandb.login(
            key=st.session_state.wandb_api_key, relogin=True
        )
        if is_wandb_logged_in:
            st.session_state.is_authenticated = True
            st.success("Logged in to Wandb")
        else:
            st.error("Failed to log in to Wandb")
