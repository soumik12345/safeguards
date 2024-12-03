from typing import Optional

import torch
import torch.nn.functional as F
import weave
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..base import Guardrail


class PromptInjectionLlamaGuardrail(Guardrail):
    model_name: str = "meta-llama/Prompt-Guard-86M"
    max_sequence_length: int = 512
    temperature: float = 1.0
    jailbreak_score_threshold: float = 0.5
    indirect_injection_score_threshold: float = 0.5
    _tokenizer: Optional[AutoTokenizer] = None
    _model: Optional[AutoModelForSequenceClassification] = None

    def model_post_init(self, __context):
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )

    def get_class_probabilities(self, prompt):
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        scaled_logits = logits / self.temperature
        probabilities = F.softmax(scaled_logits, dim=-1)
        return probabilities

    @weave.op()
    def get_score(self, prompt: str):
        probabilities = self.get_class_probabilities(prompt)
        return {
            "jailbreak_score": probabilities[0, 2].item(),
            "indirect_injection_score": (
                probabilities[0, 1] + probabilities[0, 2]
            ).item(),
        }

    @weave.op()
    def guard(self, prompt: str):
        score = self.get_score(prompt)
        summary = ""
        if score["jailbreak_score"] > self.jailbreak_score_threshold:
            confidence = round(score["jailbreak_score"] * 100, 2)
            summary += f"Prompt is deemed to be a jailbreak attempt with {confidence}% confidence."
        if score["indirect_injection_score"] > self.indirect_injection_score_threshold:
            confidence = round(score["indirect_injection_score"] * 100, 2)
            summary += f" Prompt is deemed to be an indirect injection attempt with {confidence}% confidence."
        return {
            "safe": score["jailbreak_score"] < self.jailbreak_score_threshold
            and score["indirect_injection_score"]
            < self.indirect_injection_score_threshold,
            "summary": summary.strip(),
        }

    @weave.op()
    def predict(self, prompt: str):
        return self.guard(prompt)
