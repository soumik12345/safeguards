import asyncio
import os
from importlib import import_module

import pandas as pd
import rich
import streamlit as st
import weave
from dotenv import load_dotenv

from guardrails_genie.guardrails import GuardrailManager
from guardrails_genie.llm import OpenAIModel
from guardrails_genie.metrics import AccuracyMetric


def initialize_session_state():
    load_dotenv()
    if "weave_project_name" not in st.session_state:
        st.session_state.weave_project_name = "guardrails-genie"
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = None
    if "preview_in_app" not in st.session_state:
        st.session_state.preview_in_app = False
    if "is_dataset_published" not in st.session_state:
        st.session_state.is_dataset_published = False
    if "publish_dataset_button" not in st.session_state:
        st.session_state.publish_dataset_button = False
    if "dataset_ref" not in st.session_state:
        st.session_state.dataset_ref = None


initialize_session_state()
st.title(":material/monitoring: Evaluation")

weave_project_name = st.sidebar.text_input(
    "Weave project name", value=st.session_state.weave_project_name
)
st.session_state.weave_project_name = weave_project_name
if st.session_state.weave_project_name != "":
    weave.init(project_name=st.session_state.weave_project_name)

uploaded_file = st.sidebar.file_uploader(
    "Upload the evaluation dataset as a CSV file", type="csv"
)
st.session_state.uploaded_file = uploaded_file

if st.session_state.uploaded_file is not None:
    dataset_name = st.sidebar.text_input("Evaluation dataset name", value=None)
    st.session_state.dataset_name = dataset_name
    preview_in_app = st.sidebar.toggle("Preview in app", value=False)
    st.session_state.preview_in_app = preview_in_app
    publish_dataset_button = st.sidebar.button("Publish dataset")
    st.session_state.publish_dataset_button = publish_dataset_button

    if (
        st.session_state.publish_dataset_button
        and (
            st.session_state.dataset_name is not None
            and st.session_state.dataset_name != ""
        )
    ):
        
        with st.expander("Evaluation Dataset Preview", expanded=True):
            dataframe = pd.read_csv(st.session_state.uploaded_file)
            data_list = dataframe.to_dict(orient="records")

            dataset = weave.Dataset(name=st.session_state.dataset_name, rows=data_list)
            st.session_state.dataset_ref = weave.publish(dataset)

            entity = st.session_state.dataset_ref.entity
            project = st.session_state.dataset_ref.project
            dataset_name = st.session_state.dataset_name
            digest = st.session_state.dataset_ref._digest
            dataset_url = f"https://wandb.ai/{entity}/{project}/weave/objects/{dataset_name}/versions/{digest}"
            st.markdown(
                f"Dataset published to [**Weave**]({dataset_url})"
            )

            if preview_in_app:
                st.dataframe(dataframe.head(20))
                if len(dataframe) > 20:
                    st.markdown(
                        f"⚠️ Dataset is too large to preview in app, please explore in the [**Weave UI**]({dataset_url})"
                    )

        st.session_state.is_dataset_published = True
    
    if st.session_state.is_dataset_published:
        st.write("Maza Ayega")
