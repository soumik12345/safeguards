from typing import Optional

import weave

from ...base import Guardrail
from .huggingface_classifier_guardrail import (
    PromptInjectionHuggingFaceClassifierGuardrail,
)
from .llama_prompt_guardrail import PromptInjectionLlamaGuardrail


class PromptInjectionClassifierGuardrail(Guardrail):
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
        return self.classifier_guardrail.guard(prompt)

    @weave.op()
    def predict(self, prompt: str):
        return self.guard(prompt)
