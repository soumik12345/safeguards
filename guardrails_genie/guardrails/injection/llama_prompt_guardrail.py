from typing import Optional

import torch
import torch.nn.functional as F
import weave
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..base import Guardrail


class PromptInjectionLlamaGuardrail(Guardrail):
    """
    A guardrail class designed to detect and mitigate prompt injection attacks
    using a pre-trained language model. This class leverages a sequence
    classification model to evaluate prompts for potential security threats
    such as jailbreak attempts and indirect injection attempts.

    Attributes:
        model_name (str): The name of the pre-trained model used for sequence
            classification.
        max_sequence_length (int): The maximum length of the input sequence
            for the tokenizer.
        temperature (float): A scaling factor for the model's logits to
            control the randomness of predictions.
        jailbreak_score_threshold (float): The threshold above which a prompt
            is considered a jailbreak attempt.
        indirect_injection_score_threshold (float): The threshold above which
            a prompt is considered an indirect injection attempt.
    """

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

    """
    Analyzes a given prompt to determine its safety by evaluating the likelihood
    of it being a jailbreak or indirect injection attempt.

    This function utilizes the `get_score` method to obtain the probabilities
    associated with the prompt being a jailbreak or indirect injection attempt.
    It then compares these probabilities against predefined thresholds to assess
    the prompt's safety. If the `jailbreak_score` exceeds the `jailbreak_score_threshold`,
    the prompt is flagged as a potential jailbreak attempt, and a confidence level
    is calculated and included in the summary. Similarly, if the `indirect_injection_score`
    surpasses the `indirect_injection_score_threshold`, the prompt is flagged as a potential
    indirect injection attempt, with its confidence level also included in the summary.

    Returns a dictionary containing:
        - "safe": A boolean indicating whether the prompt is considered safe
          (i.e., both scores are below their respective thresholds).
        - "summary": A string summarizing the findings, including confidence levels
          for any detected threats.
    """

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
