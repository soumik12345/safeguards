import asyncio
from typing import Any

import weave
from guardrails import Guard
from guardrails.hub import SecretsPresent
from llm_guard.input_scanners import Secrets
from llm_guard.util import configure_logger

from guardrails_genie.guardrails import GuardrailManager
from guardrails_genie.guardrails.base import Guardrail
from guardrails_genie.guardrails.secrets_detection import (
    SecretsDetectionGuardrail,
    SecretsDetectionResponse,
    SecretsDetectionSimpleResponse,
)
from guardrails_genie.metrics import AccuracyMetric

logger = configure_logger(log_level="ERROR")


class GuardrailsAISecretsDetector(Guardrail):
    """
    A class to detect secrets using Guardrails AI.

    Attributes:
        validator (Any): The validator used for detecting secrets.
    """

    validator: Any

    def __init__(self):
        """
        Initializes the GuardrailsAISecretsDetector with a validator.
        """
        validator = Guard().use(SecretsPresent, on_fail="fix")
        super().__init__(validator=validator)

    def scan(self, text: str) -> dict:
        """
        Scans the given text for secrets.

        Args:
            text (str): The text to scan for secrets.

        Returns:
            dict: A dictionary containing the scan results.
        """
        response = self.validator.validate(text)
        if response.validation_summaries:
            summary = response.validation_summaries[0]
            return {
                "has_secret": True,
                "detected_secrets": {
                    str(k): v
                    for k, v in enumerate(
                        summary.failure_reason.splitlines()[1:], start=1
                    )
                },
                "explanation": summary.failure_reason,
                "modified_prompt": response.validated_output,
                "risk_score": 1.0,
            }
        else:
            return {
                "has_secret": False,
                "detected_secrets": None,
                "explanation": "No secrets detected in the text.",
                "modified_prompt": response.validated_output,
                "risk_score": 0.0,
            }

    @weave.op
    def guard(
        self,
        prompt: str,
        return_detected_secrets: bool = True,
        **kwargs,
    ) -> SecretsDetectionResponse | SecretsDetectionResponse:
        """
        Guards the given prompt by scanning for secrets.

        Args:
            prompt (str): The prompt to scan for secrets.
            return_detected_secrets (bool): Whether to return detected secrets.

        Returns:
            SecretsDetectionResponse | SecretsDetectionSimpleResponse: The response after scanning for secrets.
        """
        results = self.scan(prompt)

        if return_detected_secrets:
            return SecretsDetectionResponse(
                contains_secrets=results["has_secret"],
                detected_secrets=results["detected_secrets"],
                explanation=results["explanation"],
                redacted_text=results["modified_prompt"],
                risk_score=results["risk_score"],
            )
        else:
            return SecretsDetectionSimpleResponse(
                contains_secrets=not results["has_secret"],
                explanation=results["explanation"],
                redacted_text=results["modified_prompt"],
                risk_score=results["risk_score"],
            )


class LLMGuardSecretsDetector(Guardrail):
    """
    A class to detect secrets using LLM Guard.

    Attributes:
        validator (Any): The validator used for detecting secrets.
    """

    validator: Any

    def __init__(self):
        """
        Initializes the LLMGuardSecretsDetector with a validator.
        """
        validator = Secrets(redact_mode="all")
        super().__init__(validator=validator)

    def scan(self, text: str) -> dict:
        """
        Scans the given text for secrets.

        Args:
            text (str): The text to scan for secrets.

        Returns:
            dict: A dictionary containing the scan results.
        """
        sanitized_prompt, is_valid, risk_score = self.validator.scan(text)
        if is_valid:
            return {
                "has_secret": not is_valid,
                "detected_secrets": None,
                "explanation": "No secrets detected in the text.",
                "modified_prompt": sanitized_prompt,
                "risk_score": risk_score,
            }
        else:
            return {
                "has_secret": not is_valid,
                "detected_secrets": {},
                "explanation": "This library does not return detected secrets.",
                "modified_prompt": sanitized_prompt,
                "risk_score": risk_score,
            }

    @weave.op
    def guard(
        self,
        prompt: str,
        return_detected_secrets: bool = True,
        **kwargs,
    ) -> SecretsDetectionResponse | SecretsDetectionResponse:
        """
        Guards the given prompt by scanning for secrets.

        Args:
            prompt (str): The prompt to scan for secrets.
            return_detected_secrets (bool): Whether to return detected secrets.

        Returns:
            SecretsDetectionResponse | SecretsDetectionSimpleResponse: The response after scanning for secrets.
        """
        results = self.scan(prompt)
        if return_detected_secrets:
            return SecretsDetectionResponse(
                contains_secrets=results["has_secret"],
                detected_secrets=results["detected_secrets"],
                explanation=results["explanation"],
                redacted_text=results["modified_prompt"],
                risk_score=results["risk_score"],
            )
        else:
            return SecretsDetectionSimpleResponse(
                contains_secrets=not results["has_secret"],
                explanation=results["explanation"],
                redacted_text=results["modified_prompt"],
                risk_score=results["risk_score"],
            )


def main():
    """
    Main function to initialize and evaluate the secrets detectors.
    """
    client = weave.init("parambharat/secrets-detection")
    dataset = weave.ref("secrets-detection-benchmark:latest").get()
    llm_guard_guardrail = LLMGuardSecretsDetector()
    guardrails_ai_guardrail = GuardrailsAISecretsDetector()
    guardrails_genie_guardrail = SecretsDetectionGuardrail()

    all_guards = [
        llm_guard_guardrail,
        guardrails_ai_guardrail,
        guardrails_genie_guardrail,
    ]
    evaluation = weave.Evaluation(
        dataset=dataset.rows,
        scorers=[AccuracyMetric()],
    )

    for guard in all_guards:
        name = guard.__class__.__name__
        guardrail_manager = GuardrailManager(
            guardrails=[
                guard,
            ]
        )

        results = asyncio.run(
            evaluation.evaluate(
                guardrail_manager,
                __weave={"display_name": f"{name}"},
            )
        )
        print(results)


if __name__ == "__main__":
    main()
