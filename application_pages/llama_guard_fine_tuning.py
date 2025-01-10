import os

import streamlit as st

from safeguards.train.llama_guard import DatasetArgs, LlamaGuardFineTuner


def initialize_session_state():
    st.session_state.llama_guard_fine_tuner = LlamaGuardFineTuner(
        wandb_project=os.getenv("WANDB_PROJECT_NAME"),
        wandb_entity=os.getenv("WANDB_ENTITY_NAME"),
        streamlit_mode=True,
    )
    if "dataset_address" not in st.session_state:
        st.session_state.dataset_address = ""
    if "train_dataset_range" not in st.session_state:
        st.session_state.train_dataset_range = 0
    if "test_dataset_range" not in st.session_state:
        st.session_state.test_dataset_range = 0
    if "load_fine_tuner_button" not in st.session_state:
        st.session_state.load_fine_tuner_button = False
    if "is_fine_tuner_loaded" not in st.session_state:
        st.session_state.is_fine_tuner_loaded = False
    if "model_name" not in st.session_state:
        st.session_state.model_name = ""
    if "preview_dataset" not in st.session_state:
        st.session_state.preview_dataset = False
    if "evaluate_model" not in st.session_state:
        st.session_state.evaluate_model = False
    if "evaluation_batch_size" not in st.session_state:
        st.session_state.evaluation_batch_size = None
    if "evaluation_temperature" not in st.session_state:
        st.session_state.evaluation_temperature = None
    if "checkpoint" not in st.session_state:
        st.session_state.checkpoint = None
    if "eval_batch_size" not in st.session_state:
        st.session_state.eval_batch_size = 32
    if "eval_positive_label" not in st.session_state:
        st.session_state.eval_positive_label = 2
    if "eval_temperature" not in st.session_state:
        st.session_state.eval_temperature = 1.0


initialize_session_state()
st.title(":material/star: Fine-Tune LLama Guard")

dataset_address = st.sidebar.text_input("Dataset Address", value="")
st.session_state.dataset_address = dataset_address

if st.session_state.dataset_address != "":
    train_dataset_range = st.sidebar.number_input(
        "Train Dataset Range", value=0, min_value=0, max_value=252956
    )
    test_dataset_range = st.sidebar.number_input(
        "Test Dataset Range", value=0, min_value=0, max_value=63240
    )
    st.session_state.train_dataset_range = train_dataset_range
    st.session_state.test_dataset_range = test_dataset_range

    model_name = st.sidebar.text_input(
        label="Model Name", value="meta-llama/Prompt-Guard-86M"
    )
    st.session_state.model_name = model_name

    checkpoint = st.sidebar.text_input(label="Fine-tuned Model Checkpoint", value="")
    st.session_state.checkpoint = checkpoint

    preview_dataset = st.sidebar.toggle("Preview Dataset")
    st.session_state.preview_dataset = preview_dataset

    evaluate_model = st.sidebar.toggle("Evaluate Model")
    st.session_state.evaluate_model = evaluate_model

    if st.session_state.evaluate_model:
        eval_batch_size = st.sidebar.slider(
            label="Eval Batch Size", min_value=16, max_value=1024, value=32
        )
        st.session_state.eval_batch_size = eval_batch_size

        eval_positive_label = st.sidebar.number_input("EVal Positive Label", value=2)
        st.session_state.eval_positive_label = eval_positive_label

        eval_temperature = st.sidebar.slider(
            label="Eval Temperature", min_value=0.0, max_value=5.0, value=1.0
        )
        st.session_state.eval_temperature = eval_temperature

    load_fine_tuner_button = st.sidebar.button("Load Fine-Tuner")
    st.session_state.load_fine_tuner_button = load_fine_tuner_button

    if st.session_state.load_fine_tuner_button:
        with st.status("Loading Fine-Tuner"):
            st.session_state.llama_guard_fine_tuner.load_dataset(
                DatasetArgs(
                    dataset_address=st.session_state.dataset_address,
                    train_dataset_range=st.session_state.train_dataset_range,
                    test_dataset_range=st.session_state.test_dataset_range,
                )
            )
            st.session_state.llama_guard_fine_tuner.load_model(
                model_name=st.session_state.model_name,
                checkpoint=(
                    None
                    if st.session_state.checkpoint == ""
                    else st.session_state.checkpoint
                ),
            )
            if st.session_state.preview_dataset:
                st.session_state.llama_guard_fine_tuner.show_dataset_sample()
            if st.session_state.evaluate_model:
                st.session_state.llama_guard_fine_tuner.evaluate_model(
                    batch_size=st.session_state.eval_batch_size,
                    positive_label=st.session_state.eval_positive_label,
                    temperature=st.session_state.eval_temperature,
                )
            st.session_state.is_fine_tuner_loaded = True
