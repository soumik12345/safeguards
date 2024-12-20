from typing import Any

import torch
import weave
from pydantic import BaseModel, Field
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer

from ...llm import OpenAIModel
from ..base import Guardrail
from .prompts import (
    PRIVILEGE_ESCALATION_SYSTEM_PROMPT,
    PRIVILEGE_ESCALATION_USER_PROMPT,
)


class OpenAIPrivEscResponse(BaseModel):
    is_priv_esc: bool = Field(
        ...,
        description="Whether the prompt is a privilege escalation prompt or not. True if it is, False otherwise.",
    )
    reason: str = Field(
        ...,
        description="Reasoning for why the prompt is a privilege escalation prompt or not.",
    )


class OpenAIPrivilegeEscalationGuardrail(Guardrail):
    llm_model: OpenAIModel

    @weave.op()
    def guard(self, prompt: str, **kwargs):
        chat_completion = self.llm_model.predict(
            user_prompts=PRIVILEGE_ESCALATION_USER_PROMPT.format(prompt=prompt),
            system_prompt=PRIVILEGE_ESCALATION_SYSTEM_PROMPT,
            response_format=OpenAIPrivEscResponse,
        )
        response = chat_completion.choices[0].message.parsed

        return {
            "safe": not response.is_priv_esc,
            "summary": response.reason,
        }

    @weave.op()
    def predict(self, prompt: str) -> dict:
        return self.guard(prompt)


class SQLInjectionGuardrail(Guardrail):
    """
    Guardrail to detect SQL injection attacks after the prompt has been executed, i.e,
    the LLM has created a SQL query based on the user's prompt.
    """

    model_name: str = "cssupport/mobilebert-sql-injection-detect"

    def model_post_init(self, __context: Any) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = MobileBertTokenizer.from_pretrained(self.model_name)
        self.model = MobileBertForSequenceClassification.from_pretrained(
            self.model_name
        )
        self.model.to(device)
        self.model.eval()

    def validate_sql_injection(self, text) -> int:
        inputs = self.tokenizer(
            text, padding=False, truncation=True, return_tensors="pt", max_length=512
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        return predicted_class

    @weave.op()
    def guard(self, prompt: str) -> dict:
        predicted_class, _ = self.validate_sql_injection(prompt)
        return {
            "safe": predicted_class == 0,
            "summary": f"The prompt is {'' if predicted_class == 0 else 'not '}a SQL injection attack.",
        }

    @weave.op()
    def predict(self, prompt: str) -> dict:
        return self.guard(prompt)
