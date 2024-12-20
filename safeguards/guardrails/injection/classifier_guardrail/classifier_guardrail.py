from typing import Optional

import weave

from ...base import Guardrail
from .huggingface_classifier_guardrail import (
    PromptInjectionHuggingFaceClassifierGuardrail,
)
from .llama_prompt_guardrail import PromptInjectionLlamaGuardrail


class PromptInjectionClassifierGuardrail(Guardrail):
    """
    A guardrail class for handling prompt injection using classifier models.

    This class extends the base Guardrail class and is designed to prevent
    prompt injection attacks by utilizing a classifier model. It dynamically
    selects between different classifier guardrails based on the specified
    model name. The class supports two types of classifier guardrails:
    PromptInjectionLlamaGuardrail and PromptInjectionHuggingFaceClassifierGuardrail.

    Attributes:
        model_name (str): The name of the model to be used for classification.
        checkpoint (Optional[str]): An optional checkpoint for the model.
        classifier_guardrail (Optional[Guardrail]): The specific guardrail
            instance used for classification, initialized during post-init.

    Methods:
        model_post_init(__context):
            Initializes the classifier_guardrail attribute based on the
            model_name. If the model_name is "meta-llama/Prompt-Guard-86M",
            it uses PromptInjectionLlamaGuardrail; otherwise, it defaults to
            PromptInjectionHuggingFaceClassifierGuardrail.

        guard(prompt: str):
            Applies the guardrail to the given prompt to prevent injection.

        predict(prompt: str):
            A wrapper around the guard method to provide prediction capability
            for the given prompt.
    """
    model_name: str
    checkpoint: Optional[str] = None
    classifier_guardrail: Optional[Guardrail] = None

    def model_post_init(self, __context):
        if self.classifier_guardrail is None:
            self.classifier_guardrail = (
                PromptInjectionLlamaGuardrail(
                    model_name=self.model_name, checkpoint=self.checkpoint
                )
                if self.model_name == "meta-llama/Prompt-Guard-86M"
                else PromptInjectionHuggingFaceClassifierGuardrail(
                    model_name=self.model_name, checkpoint=self.checkpoint
                )
            )

    @weave.op()
    def guard(self, prompt: str):
        """
        Applies the classifier guardrail to the given prompt to prevent injection.

        This method utilizes the classifier_guardrail attribute, which is an instance
        of either PromptInjectionLlamaGuardrail or PromptInjectionHuggingFaceClassifierGuardrail,
        to analyze the provided prompt and determine if it is safe or potentially harmful.

        Args:
            prompt (str): The input prompt to be evaluated by the guardrail.

        Returns:
            dict: A dictionary containing the result of the guardrail evaluation,
                  indicating whether the prompt is safe or not.
        """
        return self.classifier_guardrail.guard(prompt)

    @weave.op()
    def predict(self, prompt: str):
        """
        Provides prediction capability for the given prompt by applying the guardrail.

        This method is a wrapper around the guard method, allowing for a more intuitive
        interface for evaluating prompts. It calls the guard method to perform the
        actual evaluation.

        Args:
            prompt (str): The input prompt to be evaluated by the guardrail.

        Returns:
            dict: A dictionary containing the result of the guardrail evaluation,
                  indicating whether the prompt is safe or not.
        """
        return self.guard(prompt)
