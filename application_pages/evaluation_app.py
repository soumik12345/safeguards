import asyncio
from importlib import import_module

import pandas as pd
import streamlit as st
import weave
from dotenv import load_dotenv

from guardrails_genie.guardrails import GuardrailManager
from guardrails_genie.llm import OpenAIModel
from guardrails_genie.metrics import AccuracyMetric

load_dotenv()
weave.init(project_name="guardrails-genie")


def initialize_session_state():
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = ""
    if "preview_in_app" not in st.session_state:
        st.session_state.preview_in_app = False
    if "dataset_ref" not in st.session_state:
        st.session_state.dataset_ref = None
    if "dataset_previewed" not in st.session_state:
        st.session_state.dataset_previewed = False
    if "guardrail_names" not in st.session_state:
        st.session_state.guardrail_names = []
    if "guardrails" not in st.session_state:
        st.session_state.guardrails = []
    if "start_evaluation" not in st.session_state:
        st.session_state.start_evaluation = False
    if "evaluation_summary" not in st.session_state:
        st.session_state.evaluation_summary = None
    if "guardrail_manager" not in st.session_state:
        st.session_state.guardrail_manager = None


def initialize_guardrail():
    guardrails = []
    for guardrail_name in st.session_state.guardrail_names:
        if guardrail_name == "PromptInjectionSurveyGuardrail":
            survey_guardrail_model = st.sidebar.selectbox(
                "Survey Guardrail LLM", ["", "gpt-4o-mini", "gpt-4o"]
            )
            if survey_guardrail_model:
                guardrails.append(
                    getattr(
                        import_module("guardrails_genie.guardrails"),
                        guardrail_name,
                    )(llm_model=OpenAIModel(model_name=survey_guardrail_model))
                )
        else:
            guardrails.append(
                getattr(import_module("guardrails_genie.guardrails"), guardrail_name)()
            )
    st.session_state.guardrails = guardrails
    st.session_state.guardrail_manager = GuardrailManager(guardrails=guardrails)


initialize_session_state()
st.title(":material/monitoring: Evaluation")

uploaded_file = st.sidebar.file_uploader(
    "Upload the evaluation dataset as a CSV file", type="csv"
)
st.session_state.uploaded_file = uploaded_file
dataset_name = st.sidebar.text_input("Evaluation dataset name", value="")
st.session_state.dataset_name = dataset_name
preview_in_app = st.sidebar.toggle("Preview in app", value=False)
st.session_state.preview_in_app = preview_in_app

if st.session_state.uploaded_file is not None and st.session_state.dataset_name != "":
    with st.expander("Evaluation Dataset Preview", expanded=True):
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

        if preview_in_app:
            st.dataframe(dataframe)

        st.session_state.dataset_previewed = True

if st.session_state.dataset_previewed:
    guardrail_names = st.sidebar.multiselect(
        "Select Guardrails",
        options=[
            cls_name
            for cls_name, cls_obj in vars(
                import_module("guardrails_genie.guardrails")
            ).items()
            if isinstance(cls_obj, type) and cls_name != "GuardrailManager"
        ],
    )
    st.session_state.guardrail_names = guardrail_names

    if st.session_state.guardrail_names != []:
        initialize_guardrail()
        if st.session_state.guardrail_manager is not None:
            if st.sidebar.button("Start Evaluation"):
                st.session_state.start_evaluation = True
            if st.session_state.start_evaluation:
                evaluation = weave.Evaluation(
                    dataset=st.session_state.dataset_ref,
                    scorers=[AccuracyMetric()],
                    streamlit_mode=True,
                )
                with st.expander("Evaluation Results", expanded=True):
                    evaluation_summary, call = asyncio.run(
                        evaluation.evaluate.call(
                            evaluation, st.session_state.guardrail_manager
                        )
                    )
                    st.markdown(f"[Explore evaluation in Weave]({call.ui_url})")
                    st.write(evaluation_summary)
                st.session_state.evaluation_summary = evaluation_summary
                st.session_state.start_evaluation = False
