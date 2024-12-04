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
    if "load_dataset_button" not in st.session_state:
        st.session_state.load_dataset_button = False


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
    load_dataset_button = st.sidebar.button("Load Dataset")
    st.session_state.load_dataset_button = load_dataset_button
    if load_dataset_button:
        with st.status("Dataset Arguments"):
            dataset_args = DatasetArgs(
                dataset_address=st.session_state.dataset_address,
                train_dataset_range=st.session_state.train_dataset_range,
                test_dataset_range=st.session_state.test_dataset_range,
            )
            st.session_state.llama_guard_fine_tuner.load_dataset(dataset_args)
            st.session_state.llama_guard_fine_tuner.show_dataset_sample()
