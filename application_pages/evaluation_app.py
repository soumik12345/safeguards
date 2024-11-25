import pandas as pd
import streamlit as st
import weave
from dotenv import load_dotenv

load_dotenv()
weave.init(project_name="guardrails-genie")

st.title(":material/monitoring: Evaluation")

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = ""
if "visualize_in_app" not in st.session_state:
    st.session_state.visualize_in_app = False
if "dataset_ref" not in st.session_state:
    st.session_state.dataset_ref = None

uploaded_file = st.sidebar.file_uploader(
    "Upload the evaluation dataset as a CSV file", type="csv"
)
st.session_state.uploaded_file = uploaded_file
dataset_name = st.sidebar.text_input("Evaluation dataset name", value="")
st.session_state.dataset_name = dataset_name
visualize_in_app = st.sidebar.toggle("Visualize in app", value=False)
st.session_state.visualize_in_app = visualize_in_app

if st.session_state.uploaded_file is not None and st.session_state.dataset_name != "":
    with st.expander("Evaluation Dataset Preview"):
        dataframe = pd.read_csv(st.session_state.uploaded_file)
        data_list = dataframe.to_dict(orient="records")

        dataset = weave.Dataset(name=st.session_state.dataset_name, rows=data_list)
        st.session_state.dataset_ref = weave.publish(dataset)

        entity = st.session_state.dataset_ref.entity
        project = st.session_state.dataset_ref.project
        dataset_name = st.session_state.dataset_name
        digest = st.session_state.dataset_ref._digest
        st.markdown(
            f"Dataset published to [**Weave**](https://wandb.ai/{entity}/{project}/weave/objects/{dataset_name}/versions/{digest})"
        )

        if visualize_in_app:
            st.dataframe(dataframe)
