import asyncio
import os
import time
from importlib import import_module

import pandas as pd
import rich
import streamlit as st
import weave
from dotenv import load_dotenv

from guardrails_genie.guardrails import GuardrailManager
from guardrails_genie.llm import OpenAIModel
from guardrails_genie.metrics import AccuracyMetric
from guardrails_genie.utils import EvaluationCallManager


def initialize_session_state():
    load_dotenv()
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
    if "evaluation_name" not in st.session_state:
        st.session_state.evaluation_name = ""
    if "show_result_table" not in st.session_state:
        st.session_state.show_result_table = False
    if "weave_client" not in st.session_state:
        st.session_state.weave_client = weave.init(
            project_name=os.getenv("WEAVE_PROJECT")
        )
    if "evaluation_call_manager" not in st.session_state:
        st.session_state.evaluation_call_manager = None
    if "call_id" not in st.session_state:
        st.session_state.call_id = None
    if "llama_guardrail_checkpoint" not in st.session_state:
        st.session_state.llama_guardrail_checkpoint = None


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
        elif guardrail_name == "PromptInjectionClassifierGuardrail":
            classifier_model_name = st.sidebar.selectbox(
                "Classifier Guardrail Model",
                [
                    "",
                    "ProtectAI/deberta-v3-base-prompt-injection-v2",
                    "wandb://geekyrakshit/guardrails-genie/model-6rwqup9b:v3",
                ],
            )
            if classifier_model_name:
                st.session_state.guardrails.append(
                    getattr(
                        import_module("guardrails_genie.guardrails"),
                        guardrail_name,
                    )(model_name=classifier_model_name)
                )
        elif guardrail_name == "PromptInjectionLlamaGuardrail":
            llama_guardrail_checkpoint = st.sidebar.text_input(
                "Llama Guardrail Checkpoint",
                value=None,
            )
            st.session_state.llama_guardrail_checkpoint = llama_guardrail_checkpoint
            if st.session_state.llama_guardrail_checkpoint is not None:
                st.session_state.guardrails.append(
                    getattr(
                        import_module("guardrails_genie.guardrails"),
                        guardrail_name,
                    )(checkpoint=st.session_state.llama_guardrail_checkpoint)
                )
        else:
            st.session_state.guardrails.append(
                getattr(
                    import_module("guardrails_genie.guardrails"),
                    guardrail_name,
                )()
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
        evaluation_name = st.sidebar.text_input("Evaluation name", value="")
        st.session_state.evaluation_name = evaluation_name
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
                            evaluation,
                            st.session_state.guardrail_manager,
                            __weave={
                                "display_name": "Evaluation.evaluate:"
                                + st.session_state.evaluation_name
                            },
                        )
                    )
                    x_axis = list(evaluation_summary["AccuracyMetric"].keys())
                    y_axis = [
                        evaluation_summary["AccuracyMetric"][x_axis_item]
                        for x_axis_item in x_axis
                    ]
                    st.bar_chart(
                        pd.DataFrame({"Metric": x_axis, "Score": y_axis}),
                        x="Metric",
                        y="Score",
                    )
                    st.session_state.evaluation_summary = evaluation_summary
                    st.session_state.call_id = call.id
                    st.session_state.start_evaluation = False

                    if not st.session_state.start_evaluation:
                        time.sleep(5)
                        st.session_state.evaluation_call_manager = (
                            EvaluationCallManager(
                                entity="geekyrakshit",
                                project="guardrails-genie",
                                call_id=st.session_state.call_id,
                            )
                        )
                        for guardrail_name in st.session_state.guardrail_names:
                            st.session_state.evaluation_call_manager.call_list.append(
                                {
                                    "guardrail_name": guardrail_name,
                                    "calls": st.session_state.evaluation_call_manager.collect_guardrail_guard_calls_from_eval(),
                                }
                            )
                            rich.print(
                                st.session_state.evaluation_call_manager.call_list
                            )
                        st.dataframe(
                            st.session_state.evaluation_call_manager.render_calls_to_streamlit()
                        )
                        if st.session_state.evaluation_call_manager.show_warning_in_app:
                            st.warning(
                                f"Only {st.session_state.evaluation_call_manager.max_count} calls can be shown in the app."
                            )
                        st.markdown(
                            f"Explore the entire evaluation trace table in [Weave]({call.ui_url})"
                        )
                    st.session_state.evaluation_call_manager = None
