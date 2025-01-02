import importlib
from typing import Dict, List, Union

import pandas as pd
import streamlit as st
import weave
from dotenv import load_dotenv
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from .guardrails import GuardrailManager
from .llm import OpenAIModel

load_dotenv()


class EvaluationCallManager:
    """
    Manages the evaluation calls for a specific project and entity in Weave.

    This class is responsible for initializing and managing evaluation calls associated with a
    specific project and entity. It provides functionality to collect guardrail guard calls
    from evaluation predictions and scores, and render these calls into a structured format
    suitable for display in Streamlit.

    Args:
        entity (str): The entity name.
        project (str): The project name.
        call_id (str): The call id.
        max_count (int): The maximum number of guardrail guard calls to collect from the evaluation.
    """

    def __init__(self, entity: str, project: str, call_id: str, max_count: int = 10):
        self.base_call = weave.init(f"{entity}/{project}").get_call(call_id=call_id)
        self.max_count = max_count
        self.show_warning_in_app = False
        self.call_list = []

    def collect_guardrail_guard_calls_from_eval(self):
        """
        Collects guardrail guard calls from evaluation predictions and scores.

        This function iterates through the children calls of the base evaluation call,
        extracting relevant guardrail guard calls and their associated scores. It stops
        collecting calls if it encounters an "Evaluation.summarize" operation or if the
        maximum count of guardrail guard calls is reached. The collected calls are stored
        in a list of dictionaries, each containing the input prompt, outputs, and score.

        Returns:
            list: A list of dictionaries, each containing:
                - input_prompt (str): The input prompt for the guard call.
                - outputs (dict): The outputs of the guard call.
                - score (dict): The score of the guard call.
        """
        guard_calls, count = [], 0
        for eval_predict_and_score_call in self.base_call.children():
            if "Evaluation.summarize" in eval_predict_and_score_call._op_name:
                break
            guardrail_predict_call = eval_predict_and_score_call.children()[0]
            guard_call = guardrail_predict_call.children()[0]
            score_call = eval_predict_and_score_call.children()[1]
            guard_calls.append(
                {
                    "input_prompt": str(guard_call.inputs["prompt"]),
                    "outputs": dict(guard_call.output),
                    "score": dict(score_call.output),
                }
            )
            count += 1
            if count >= self.max_count:
                self.show_warning_in_app = True
                break
        return guard_calls

    def render_calls_to_streamlit(self):
        """
        Renders the collected guardrail guard calls into a pandas DataFrame suitable for
        display in Streamlit.

        This function processes the collected guardrail guard calls stored in `self.call_list` and
        organizes them into a dictionary format that can be easily converted into a pandas DataFrame.
        The DataFrame contains columns for the input prompts, the safety status of the outputs, and
        the correctness of the predictions for each guardrail.

        The structure of the DataFrame is as follows:
        - The first column contains the input prompts.
        - Subsequent columns contain the safety status and prediction correctness for each guardrail.

        Returns:
            pd.DataFrame: A DataFrame containing the input prompts, safety status, and prediction
                          correctness for each guardrail.
        """
        dataframe = {
            "input_prompt": [
                call["input_prompt"] for call in self.call_list[0]["calls"]
            ]
        }
        for guardrail_call in self.call_list:
            dataframe[guardrail_call["guardrail_name"] + ".safe"] = [
                call["outputs"]["safe"] for call in guardrail_call["calls"]
            ]
            dataframe[guardrail_call["guardrail_name"] + ".prediction_correctness"] = [
                call["score"]["correct"] for call in guardrail_call["calls"]
            ]
        return pd.DataFrame(dataframe)


class StreamlitProgressbarCallback(TrainerCallback):
    """
    StreamlitProgressbarCallback is a custom callback for the Hugging Face Trainer
    that integrates a progress bar into a Streamlit application. This class updates
    the progress bar at each training step, providing real-time feedback on the
    training process within the Streamlit interface.

    Attributes:
        progress_bar (streamlit.delta_generator.DeltaGenerator): A Streamlit progress
            bar object initialized to 0 with the text "Training".

    Methods:
        on_step_begin(args, state, control, **kwargs):
            Updates the progress bar at the beginning of each training step. The progress
            is calculated as the percentage of completed steps out of the total steps.
            The progress bar text is updated to show the current step and the total steps.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_bar = st.progress(0, text="Training")

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_step_begin(args, state, control, **kwargs)
        self.progress_bar.progress(
            (state.global_step * 100 // state.max_steps) + 1,
            text=f"Training {state.global_step} / {state.max_steps}",
        )


def initialize_guardrails_on_playground():
    """
    Initializes guardrails for the Streamlit application based on the user's selection
    from the sidebar. This function dynamically imports and configures various guardrail
    classes from the 'guardrails_genie.guardrails' module, depending on the guardrail
    names specified in the Streamlit session state.

    The function iterates over each guardrail name in 'st.session_state.guardrail_names'
    and performs the following actions based on the guardrail type:

    - For "PromptInjectionLLMGuardrail", it allows the user to select a language model
      from a dropdown and initializes the guardrail with the selected model.
    - For "PromptInjectionClassifierGuardrail", it initializes the guardrail with a
      predefined model name.
    - For "PromptInjectionLlamaGuardrail", it takes a checkpoint name input from the user
      and initializes the guardrail with the specified checkpoint.
    - For entity recognition guardrails like "PresidioEntityRecognitionGuardrail",
      "RegexEntityRecognitionGuardrail", and "TransformersEntityRecognitionGuardrail",
      it provides a checkbox for the user to decide whether to anonymize entities and
      initializes the guardrail accordingly.
    - For "RestrictedTermsJudge", it provides a checkbox for anonymization and initializes
      the guardrail based on the user's choice.
    - For any other guardrail names, it initializes the guardrail with default settings.

    After initializing all guardrails, it creates a 'GuardrailManager' instance with the
    configured guardrails and stores it in the session state for further use in the
    application.
    """
    st.session_state.guardrails = []
    for guardrail_name in st.session_state.guardrail_names:
        if guardrail_name == "PromptInjectionLLMGuardrail":
            prompt_injection_llm_model = st.sidebar.selectbox(
                "Prompt Injection Guardrail LLM", ["gpt-4o-mini", "gpt-4o"]
            )
            st.session_state.prompt_injection_llm_model = prompt_injection_llm_model
            st.session_state.guardrails.append(
                getattr(
                    importlib.import_module("safeguards.guardrails"),
                    guardrail_name,
                )(
                    llm_model=OpenAIModel(
                        model_name=st.session_state.prompt_injection_llm_model
                    )
                )
            )
        elif guardrail_name == "PromptInjectionClassifierGuardrail":
            st.session_state.guardrails.append(
                getattr(
                    importlib.import_module("safeguards.guardrails"),
                    guardrail_name,
                )(model_name="ProtectAI/deberta-v3-base-prompt-injection-v2")
            )
        elif guardrail_name == "PromptInjectionLlamaGuardrail":
            prompt_injection_llama_guard_checkpoint_name = st.sidebar.text_input(
                "Checkpoint Name",
                value="wandb://geekyrakshit/guardrails-genie/ruk3f3b4-model:v8",
            )
            st.session_state.prompt_injection_llama_guard_checkpoint_name = (
                prompt_injection_llama_guard_checkpoint_name
            )
            st.session_state.guardrails.append(
                getattr(
                    importlib.import_module("safeguards.guardrails"),
                    guardrail_name,
                )(
                    checkpoint=(
                        None
                        if st.session_state.prompt_injection_llama_guard_checkpoint_name
                        == ""
                        else st.session_state.prompt_injection_llama_guard_checkpoint_name
                    )
                )
            )
        elif guardrail_name == "PresidioEntityRecognitionGuardrail":
            presidio_entity_recognition_guardrail_should_anonymize = (
                st.sidebar.checkbox(
                    "Presidio Entity Recognition Guardrail: Anonymize", value=True
                )
            )
            st.session_state.presidio_entity_recognition_guardrail_should_anonymize = (
                presidio_entity_recognition_guardrail_should_anonymize
            )
            st.session_state.guardrails.append(
                getattr(
                    importlib.import_module("safeguards.guardrails"),
                    guardrail_name,
                )(
                    should_anonymize=st.session_state.presidio_entity_recognition_guardrail_should_anonymize
                )
            )
        elif guardrail_name == "RegexEntityRecognitionGuardrail":
            regex_entity_recognition_guardrail_should_anonymize = st.sidebar.checkbox(
                "Regex Entity Recognition Guardrail: Anonymize", value=True
            )
            st.session_state.regex_entity_recognition_guardrail_should_anonymize = (
                regex_entity_recognition_guardrail_should_anonymize
            )
            st.session_state.guardrails.append(
                getattr(
                    importlib.import_module("safeguards.guardrails"),
                    guardrail_name,
                )(
                    should_anonymize=st.session_state.regex_entity_recognition_guardrail_should_anonymize
                )
            )
        elif guardrail_name == "TransformersEntityRecognitionGuardrail":
            transformers_entity_recognition_guardrail_should_anonymize = (
                st.sidebar.checkbox(
                    "Transformers Entity Recognition Guardrail: Anonymize", value=True
                )
            )
            st.session_state.transformers_entity_recognition_guardrail_should_anonymize = (
                transformers_entity_recognition_guardrail_should_anonymize
            )
            st.session_state.guardrails.append(
                getattr(
                    importlib.import_module("safeguards.guardrails"),
                    guardrail_name,
                )(
                    should_anonymize=st.session_state.transformers_entity_recognition_guardrail_should_anonymize
                )
            )
        elif guardrail_name == "RestrictedTermsJudge":
            restricted_terms_judge_should_anonymize = st.sidebar.checkbox(
                "Restricted Terms Judge: Anonymize", value=True
            )
            st.session_state.restricted_terms_judge_should_anonymize = (
                restricted_terms_judge_should_anonymize
            )
            st.session_state.guardrails.append(
                getattr(
                    importlib.import_module("safeguards.guardrails"),
                    guardrail_name,
                )(
                    should_anonymize=st.session_state.restricted_terms_judge_should_anonymize
                )
            )
        else:
            st.session_state.guardrails.append(
                getattr(
                    importlib.import_module("safeguards.guardrails"),
                    guardrail_name,
                )()
            )
    st.session_state.guardrails_manager = GuardrailManager(
        guardrails=st.session_state.guardrails
    )


def remove_class_key(d: Union[Dict, List]):
    if isinstance(d, dict):
        d.pop("__class__", None)
        for key, value in d.items():
            remove_class_key(value)
    elif isinstance(d, list):
        for item in d:
            remove_class_key(item)
    return d
