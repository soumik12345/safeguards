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
    """
    Guardrail to detect privilege escalation prompts using an OpenAI language model.

    This class uses an OpenAI language model to predict whether a given prompt
    is attempting to perform a privilege escalation. It does so by sending the
    prompt to the model along with predefined system and user prompts, and then
    analyzing the model's response.

    Attributes:
        llm_model (OpenAIModel): The language model used to predict privilege escalation.

    Methods:
        guard(prompt: str, **kwargs) -> dict:
            Analyzes the given prompt to determine if it is a privilege escalation attempt.
            Returns a dictionary with the analysis result.

        predict(prompt: str) -> dict:
            A wrapper around the guard method to provide a consistent interface.
    """
    llm_model: OpenAIModel

    @weave.op()
    def guard(self, prompt: str, **kwargs):
        """
        Analyzes the given prompt to determine if it is a privilege escalation attempt.

        This function uses an OpenAI language model to predict whether a given prompt
        is attempting to perform a privilege escalation. It sends the prompt to the model
        along with predefined system and user prompts, and then analyzes the model's response.
        The response is parsed to check if the prompt is a privilege escalation attempt and
        provides a summary of the reasoning.

        Args:
            prompt (str): The input prompt to be analyzed.
            **kwargs: Additional keyword arguments that may be passed to the model's predict method.

        Returns:
            dict: A dictionary containing the safety status and a summary of the analysis.
                  - "safe" (bool): Indicates whether the prompt is safe (True) or a privilege escalation attempt (False).
                  - "summary" (str): A summary of the reasoning behind the classification.
        """
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
        """
        A wrapper around the guard method to provide a consistent interface.

        This function calls the guard method to analyze the given prompt and determine
        if it is a privilege escalation attempt. It returns the result of the guard method.

        Args:
            prompt (str): The input prompt to be analyzed.

        Returns:
            dict: A dictionary containing the safety status and a summary of the analysis.
        """
        return self.guard(prompt)


class SQLInjectionGuardrail(Guardrail):
    """
    A guardrail class designed to detect SQL injection attacks in SQL queries
    generated by a language model (LLM) based on user prompts.

    This class utilizes a pre-trained MobileBERT model for sequence classification
    to evaluate whether a given SQL query is potentially harmful due to SQL injection.
    It leverages the model's ability to classify text sequences to determine if the
    query is safe or indicative of an injection attack.

    Attributes:
        model_name (str): The name of the pre-trained MobileBERT model used for
            SQL injection detection.

    Methods:
        model_post_init(__context: Any) -> None:
            Initializes the tokenizer and model for sequence classification,
            setting the model to evaluation mode and moving it to the appropriate
            device (CPU or GPU).

        validate_sql_injection(text: str) -> int:
            Processes the input text using the tokenizer and model to predict
            the class of the SQL query. Returns the predicted class, where 0
            indicates a safe query and 1 indicates a potential SQL injection.

        guard(prompt: str) -> dict:
            Analyzes the given prompt to determine if it results in a SQL injection
            attack. Returns a dictionary with the safety status and a summary of
            the analysis.

        predict(prompt: str) -> dict:
            A wrapper around the guard method to provide a consistent interface
            for evaluating prompts.
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
        """
        Analyzes the given prompt to determine if it results in a SQL injection attack.

        This function uses the `validate_sql_injection` method to process the input prompt
        and predict whether it is a safe query or a potential SQL injection attack. The
        prediction is based on a pre-trained MobileBERT model for sequence classification.
        The function returns a dictionary containing the safety status and a summary of
        the analysis.

        Args:
            prompt (str): The input prompt to be analyzed.

        Returns:
            dict: A dictionary with two keys:
                - "safe": A boolean indicating whether the prompt is safe (True) or a
                    SQL injection attack (False).
                - "summary": A string summarizing the analysis result, indicating whether
                    the prompt is a SQL injection attack.
        """
        predicted_class, _ = self.validate_sql_injection(prompt)
        return {
            "safe": predicted_class == 0,
            "summary": f"The prompt is {'' if predicted_class == 0 else 'not '}a SQL injection attack.",
        }

    @weave.op()
    def predict(self, prompt: str) -> dict:
        """
        A wrapper around the `guard` method to provide a consistent interface for evaluating prompts.

        This function calls the `guard` method to analyze the given prompt and determine if it
        results in a SQL injection attack. It returns the same dictionary as the `guard` method,
        containing the safety status and a summary of the analysis.

        Args:
            prompt (str): The input prompt to be evaluated.

        Returns:
            dict: A dictionary with two keys:
                - "safe": A boolean indicating whether the prompt is safe (True) or a
                    SQL injection attack (False).
                - "summary": A string summarizing the analysis result, indicating
                    whether the prompt is a SQL injection attack.
        """
        return self.guard(prompt)