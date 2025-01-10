import os

import streamlit as st
import wandb


def initialize_session_state():
    if "weave_project_name" not in st.session_state:
        st.session_state.weave_project_name = "safeguards"
    if "weave_entity_name" not in st.session_state:
        st.session_state.weave_entity_name = ""
    if "wandb_api_key" not in st.session_state:
        st.session_state.wandb_api_key = ""
    if "authenticate_button" not in st.session_state:
        st.session_state.authenticate_button = False
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False


initialize_session_state()
st.title("Safeguards: Guardrails for AI Applications")

st.write(
    """
[![Docs](https://img.shields.io/badge/documentation-online-green.svg)](https://geekyrakshit.dev/safeguards)

A comprehensive collection of guardrails for securing and validating prompts in AI applications built on top of [Weights & Biases Weave](https://wandb.me/weave). The library provides multiple types of guardrails for entity recognition, prompt injection detection, and other security measures.

## Features

- Built on top of [Weights & Biases Weave](https://wandb.me/weave) - the observability platform for AI evaluation, iteration, and monitoring.
- Multiple types of guardrails for entity recognition, prompt injection detection, and other security measures.
- [Playground](/chat_app) for testing and utilizing guardrails.
- [Evaluation](/evaluation_app) of guardrails using Weave on your own data.

## 👈 Authenticate with your W&B and OpenAI API keys to get started
"""
)

st.sidebar.markdown(
    "Get your Wandb API key from [https://wandb.ai/authorize](https://wandb.ai/authorize)"
)
weave_entity_name = st.sidebar.text_input(
    "Weave Entity Name", value=st.session_state.weave_entity_name
)
st.session_state.weave_entity_name = weave_entity_name
weave_project_name = st.sidebar.text_input(
    "Weave Project Name", value=st.session_state.weave_project_name
)
st.session_state.weave_project_name = weave_project_name
wandb_api_key = st.sidebar.text_input("Wandb API Key", value="", type="password")
st.session_state.wandb_api_key = wandb_api_key
openai_api_key = st.sidebar.text_input("OpenAI API Key", value="", type="password")
os.environ["OPENAI_API_KEY"] = openai_api_key
authenticate_button = st.sidebar.button("Authenticate")
st.session_state.authenticate_button = authenticate_button

if authenticate_button and (
    st.session_state.wandb_api_key != ""
    and st.session_state.weave_project_name != ""
    and openai_api_key != ""
):
    is_wandb_logged_in = wandb.login(key=st.session_state.wandb_api_key, relogin=True)
    if is_wandb_logged_in:
        st.session_state.is_authenticated = True
        st.sidebar.success("Logged in to Wandb")
    else:
        st.sidebar.error("Failed to log in to Wandb")
