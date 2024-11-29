import os

import streamlit as st
from dotenv import load_dotenv

from guardrails_genie.train_classifier import train_binary_classifier


def initialize_session_state():
    load_dotenv()
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = None
    if "base_model_name" not in st.session_state:
        st.session_state.base_model_name = None
    if "batch_size" not in st.session_state:
        st.session_state.batch_size = 16
    if "should_start_training" not in st.session_state:
        st.session_state.should_start_training = False
    if "training_output" not in st.session_state:
        st.session_state.training_output = None


initialize_session_state()
st.title(":material/fitness_center: Train Classifier")

dataset_name = st.sidebar.text_input("Dataset Name", value="")
st.session_state.dataset_name = dataset_name

base_model_name = st.sidebar.selectbox(
    "Base Model",
    options=[
        "distilbert/distilbert-base-uncased",
        "FacebookAI/roberta-base",
        "microsoft/deberta-v3-base",
    ],
)
st.session_state.base_model_name = base_model_name

batch_size = st.sidebar.slider(
    "Batch Size", min_value=4, max_value=256, value=16, step=4
)
st.session_state.batch_size = batch_size

train_button = st.sidebar.button("Train")
st.session_state.should_start_training = (
    train_button and st.session_state.dataset_name and st.session_state.base_model_name
)

if st.session_state.should_start_training:
    with st.expander("Training", expanded=True):
        try:
            training_output = train_binary_classifier(
                project_name=os.getenv("WANDB_PROJECT_NAME"),
                entity_name=os.getenv("WANDB_ENTITY_NAME"),
                run_name=f"{st.session_state.base_model_name}-finetuned",
                dataset_repo=st.session_state.dataset_name,
                model_name=st.session_state.base_model_name,
                batch_size=st.session_state.batch_size,
                streamlit_mode=True,
            )
            st.session_state.training_output = training_output
            st.write(training_output)
        except Exception as e:
            st.error(e)
