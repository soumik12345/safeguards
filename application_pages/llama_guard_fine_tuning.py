import streamlit as st

from guardrails_genie.train.llama_guard import DatasetArgs, LlamaGuardFineTuner


def initialize_session_state():
    st.session_state.llama_guard_fine_tuner = LlamaGuardFineTuner(streamlit_mode=True)
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

    model_name = st.sidebar.selectbox(
        "Model Name",
        ["meta-llama/Prompt-Guard-86M"],
    )
    st.session_state.model_name = model_name

    preview_dataset = st.sidebar.toggle("Preview Dataset")
    st.session_state.preview_dataset = preview_dataset

    evaluate_model = st.sidebar.toggle("Evaluate Model")
    st.session_state.evaluate_model = evaluate_model

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
                model_name=st.session_state.model_name
            )
            if st.session_state.preview_dataset:
                st.session_state.llama_guard_fine_tuner.show_dataset_sample()
            if st.session_state.evaluate_model:
                st.session_state.llama_guard_fine_tuner.evaluate_model(
                    batch_size=32,
                    temperature=3.0,
                )
            st.session_state.is_fine_tuner_loaded = True
