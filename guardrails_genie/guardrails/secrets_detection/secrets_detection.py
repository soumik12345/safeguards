import hashlib
import json
import pathlib
from enum import Enum
from typing import Optional, Union

import weave
from pydantic import BaseModel

from guardrails_genie.guardrails.base import Guardrail
from guardrails_genie.regex_model import RegexModel


def load_secrets_patterns() -> dict[str, list[str]]:
    """
    Load secret patterns from a JSONL file and return them as a dictionary.

    Returns:
        dict: A dictionary where keys are pattern names and values are lists of regex patterns.
    """
    default_patterns = {}
    patterns = (
        pathlib.Path(__file__).parent.absolute() / "secrets_patterns.jsonl"
    ).read_text()

    for pattern in patterns.splitlines():
        pattern = json.loads(pattern)
        default_patterns[pattern["name"]] = [rf"{pat}" for pat in pattern["patterns"]]
    return default_patterns


# Load default secret patterns from the JSONL file
DEFAULT_SECRETS_PATTERNS = load_secrets_patterns()


class REDACTION(str, Enum):
    """
    Enum for different types of redaction methods.
    """

    REDACT_PARTIAL = "REDACT_PARTIAL"
    REDACT_ALL = "REDACT_ALL"
    REDACT_HASH = "REDACT_HASH"
    REDACT_NONE = "REDACT_NONE"


def redact(text: str, matches: list[str], redaction_type: REDACTION) -> str:
    """
    Redact the given matches in the text based on the redaction type.

    Args:
        text (str): The input text to redact.
        matches (list[str]): List of strings to be redacted.
        redaction_type (REDACTION): The type of redaction to apply.

    Returns:
        str: The redacted text.
    """
    for match in matches:
        if redaction_type == REDACTION.REDACT_PARTIAL:
            replacement = "[REDACTED:]" + match[:2] + ".." + match[-2:] + "[:REDACTED]"
        elif redaction_type == REDACTION.REDACT_ALL:
            replacement = "[REDACTED:]" + ("*" * len(match)) + "[:REDACTED]"
        elif redaction_type == REDACTION.REDACT_HASH:
            replacement = (
                "[REDACTED:]" + hashlib.md5(match.encode()).hexdigest() + "[:REDACTED]"
            )
        else:
            replacement = match
        text = text.replace(match, replacement)
    return text


class SecretsDetectionSimpleResponse(BaseModel):
    """
    A simple response model for secrets detection.

    Attributes:
        contains_secrets (bool): Indicates if secrets were detected.
        explanation (str): Explanation of the detection result.
        redacted_text (Optional[str]): The redacted text if secrets were found.
    """

    contains_secrets: bool
    explanation: str
    redacted_text: Optional[str] = None

    @property
    def safe(self) -> bool:
        """
        Property to check if the text is safe (no secrets detected).

        Returns:
            bool: True if no secrets were detected, False otherwise.
        """
        return not self.contains_secrets


class SecretsDetectionResponse(SecretsDetectionSimpleResponse):
    """
    A detailed response model for secrets detection.

    Attributes:
        detected_secrets (dict[str, list[str]]): Dictionary of detected secrets.
    """

    detected_secrets: dict[str, list[str]]


class SecretsDetectionGuardrail(Guardrail):
    """
    A guardrail for detecting secrets in text using regex patterns.
    reference: SecretBench: A Dataset of Software Secrets
    https://arxiv.org/abs/2303.06729

    Attributes:
        regex_model (RegexModel): The regex model used for detection.
        patterns (Union[dict[str, str], dict[str, list[str]]]): The patterns used for detection.
        redaction (REDACTION): The type of redaction to apply.
    """

    regex_model: RegexModel
    patterns: Union[dict[str, str], dict[str, list[str]]] = {}
    redaction: REDACTION

    def __init__(
        self,
        use_defaults: bool = True,
        redaction: REDACTION = REDACTION.REDACT_ALL,
        **kwargs,
    ):
        """
        Initialize the SecretsDetectionGuardrail.

        Args:
            use_defaults (bool): Whether to use default patterns.
            redaction (REDACTION): The type of redaction to apply.
            **kwargs: Additional keyword arguments.
        """
        patterns = {}
        if use_defaults:
            patterns = DEFAULT_SECRETS_PATTERNS.copy()
        if kwargs.get("patterns"):
            patterns.update(kwargs["patterns"])

        regex_model = RegexModel(patterns=patterns)

        super().__init__(
            regex_model=regex_model,
            patterns=patterns,
            redaction=redaction,
        )

    @weave.op()
    def guard(
        self,
        prompt: str,
        return_detected_secrets: bool = True,
        **kwargs,
    ) -> SecretsDetectionResponse | SecretsDetectionResponse:
        """
        Check if the input prompt contains any secrets based on the regex patterns.

        Args:
            prompt (str): Input text to check for secrets.
            return_detected_secrets (bool): If True, returns detailed secrets type information.

        Returns:
            SecretsDetectionResponse or SecretsDetectionResponse: Detection results.
        """
        result = self.regex_model.check(prompt)

        explanation_parts = []
        if result.matched_patterns:
            explanation_parts.append("Found the following secrets in the text:")
            for secret_type, matches in result.matched_patterns.items():
                explanation_parts.append(f"- {secret_type}: {len(matches)} instance(s)")
        else:
            explanation_parts.append("No secrets detected in the text.")

        redacted_text = prompt
        if result.matched_patterns:
            for secret_type, matches in result.matched_patterns.items():
                redacted_text = redact(redacted_text, matches, self.redaction)

        if return_detected_secrets:
            return SecretsDetectionResponse(
                contains_secrets=not result.passed,
                detected_secrets=result.matched_patterns,
                explanation="\n".join(explanation_parts),
                redacted_text=redacted_text,
            )
        else:
            return SecretsDetectionSimpleResponse(
                contains_secrets=not result.passed,
                explanation="\n".join(explanation_parts),
                redacted_text=redacted_text,
            )
