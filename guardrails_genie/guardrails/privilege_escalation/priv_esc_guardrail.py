from typing import Optional

import torch
import weave
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.base import Pipeline

import wandb

from ..base import Guardrail


class PrivilegeEscalationGuardrail(Guardrail):
    def __init__(self):
        super().__init__()

    @weave.op()
    def guard(self, prompt: str, **kwargs):
        return {
            "is_prev_esc": True,
            "prev_esc_type": "unknown",
        }

    @weave.op()
    def predict(self, prompt: str) -> dict:
        return self.guard(prompt)
