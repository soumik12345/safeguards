import asyncio
from importlib import import_module

import pandas as pd
import streamlit as st
import weave

from guardrails_genie.guardrails import GuardrailManager
from guardrails_genie.llm import OpenAIModel
from guardrails_genie.metrics import AccuracyMetric


def initialize_session_state():
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
    if "guardrails" not in st.session_state:
        st.session_state.guardrails = []
    if "guardrail_names" not in st.session_state:
        st.session_state.guardrail_names = []
    if "start_evaluations_button" not in st.session_state:
        st.session_state.start_evaluations_button = False
    if "evaluation_name" not in st.session_state:
        st.session_state.evaluation_name = ""


def initialize_guardrails():
    st.session_state.guardrails = []
    for guardrail_name in st.session_state.guardrail_names:
        if guardrail_name == "PromptInjectionSurveyGuardrail":
            survey_guardrail_model = st.sidebar.selectbox(
                "Survey Guardrail LLM", ["", "gpt-4o-mini", "gpt-4o"]
            )
            if survey_guardrail_model:
                st.session_state.guardrails.append(
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
                ],
            )
            if classifier_model_name != "":
                st.session_state.guardrails.append(
                    getattr(
                        import_module("guardrails_genie.guardrails"),
                        guardrail_name,
                    )(model_name=classifier_model_name)
                )
        elif guardrail_name == "PresidioEntityRecognitionGuardrail":
            st.session_state.guardrails.append(
                getattr(
                    import_module("guardrails_genie.guardrails"),
                    guardrail_name,
                )(should_anonymize=True)
            )
        elif guardrail_name == "RegexEntityRecognitionGuardrail":
            st.session_state.guardrails.append(
                getattr(
                    import_module("guardrails_genie.guardrails"),
                    guardrail_name,
                )(should_anonymize=True)
            )
        elif guardrail_name == "TransformersEntityRecognitionGuardrail":
            st.session_state.guardrails.append(
                getattr(
                    import_module("guardrails_genie.guardrails"),
                    guardrail_name,
                )(should_anonymize=True)
            )
        elif guardrail_name == "RestrictedTermsJudge":
            st.session_state.guardrails.append(
                getattr(
                    import_module("guardrails_genie.guardrails"),
                    guardrail_name,
                )(should_anonymize=True)
            )
        elif guardrail_name == "PromptInjectionLlamaGuardrail":
            llama_guard_checkpoint_name = st.sidebar.text_input(
                "Checkpoint Name", value=""
            )
            st.session_state.llama_guard_checkpoint_name = llama_guard_checkpoint_name
            st.session_state.guardrails.append(
                getattr(import_module("guardrails_genie.guardrails"), guardrail_name,)(
                    checkpoint=(
                        None
                        if st.session_state.llama_guard_checkpoint_name == ""
                        else st.session_state.llama_guard_checkpoint_name
                    )
                )
            )
        else:
            st.session_state.guardrails.append(
                getattr(
                    import_module("guardrails_genie.guardrails"),
                    guardrail_name,
                )()
            )
    st.session_state.guardrails_manager = GuardrailManager(
        guardrails=st.session_state.guardrails
    )


if st.session_state.is_authenticated:
    weave.init(
        project_name=f"{st.session_state.weave_entity_name}/{st.session_state.weave_project_name}"
    )
    initialize_session_state()
    st.title(":material/monitoring: Evaluation")

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

        if st.session_state.publish_dataset_button and (
            st.session_state.dataset_name is not None
            and st.session_state.dataset_name != ""
        ):

            with st.expander("Evaluation Dataset Preview", expanded=True):
                dataframe = pd.read_csv(st.session_state.uploaded_file)
                data_list = dataframe.to_dict(orient="records")

                dataset = weave.Dataset(
                    name=st.session_state.dataset_name, rows=data_list
                )
                st.session_state.dataset_ref = weave.publish(dataset)

                entity = st.session_state.dataset_ref.entity
                project = st.session_state.dataset_ref.project
                dataset_name = st.session_state.dataset_name
                digest = st.session_state.dataset_ref._digest
                dataset_url = f"https://wandb.ai/{entity}/{project}/weave/objects/{dataset_name}/versions/{digest}"
                st.markdown(f"Dataset published to [**Weave**]({dataset_url})")

                if preview_in_app:
                    st.dataframe(dataframe.head(20))
                    if len(dataframe) > 20:
                        st.markdown(
                            f"⚠️ Dataset is too large to preview in app, please explore in the [**Weave UI**]({dataset_url})"
                        )

            st.session_state.is_dataset_published = True

        if st.session_state.is_dataset_published:
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

            initialize_guardrails()
            evaluation_name = st.sidebar.text_input("Evaluation Name", value="")
            st.session_state.evaluation_name = evaluation_name

            start_evaluations_button = st.sidebar.button("Start Evaluations")
            st.session_state.start_evaluations_button = start_evaluations_button
            if st.session_state.start_evaluations_button:
                # st.write(len(st.session_state.guardrails))
                try:
                    evaluation = weave.Evaluation(
                        dataset=st.session_state.dataset_ref,
                        scorers=[AccuracyMetric()],
                        streamlit_mode=True,
                    )
                except Exception as e:
                    evaluation = weave.Evaluation(
                        dataset=st.session_state.dataset_ref,
                        scorers=[AccuracyMetric()],
                    )
                with st.expander("Evaluation Results", expanded=True):
                    evaluation_summary, call = asyncio.run(
                        evaluation.evaluate.call(
                            evaluation,
                            GuardrailManager(guardrails=st.session_state.guardrails),
                            __weave={
                                "display_name": (
                                    "Evaluation.evaluate"
                                    if st.session_state.evaluation_name == ""
                                    else "Evaluation.evaluate:"
                                    + st.session_state.evaluation_name
                                )
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
                    st.markdown(
                        f"Explore the entire evaluation trace table in [Weave]({call.ui_url})"
                    )
else:
    st.warning("Please authenticate your WandB account to use this feature.")
