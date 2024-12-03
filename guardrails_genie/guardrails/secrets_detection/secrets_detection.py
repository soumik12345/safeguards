import hashlib
import json
import pathlib
from enum import Enum
from typing import Union, Optional

import weave
from pydantic import BaseModel

from guardrails_genie.guardrails.base import Guardrail
from guardrails_genie.regex_model import RegexModel


def load_secrets_patterns():
    default_patterns = {}
    patterns = (
        pathlib.Path(__file__).parent.absolute() / "secrets_patterns.jsonl"
    ).read_text()

    for pattern in patterns.splitlines():
        pattern = json.loads(pattern)
        default_patterns[pattern["name"]] = [rf"{pat}" for pat in pattern["patterns"]]
    return default_patterns


DEFAULT_SECRETS_PATTERNS = load_secrets_patterns()


class REDACTION(str, Enum):
    REDACT_PARTIAL = "REDACT_PARTIAL"
    REDACT_ALL = "REDACT_ALL"
    REDACT_HASH = "REDACT_HASH"
    REDACT_NONE = "REDACT_NONE"


def redact(text: str, matches: list[str], redaction_type: REDACTION) -> str:
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
    contains_secrets: bool
    explanation: str
    redacted_text: Optional[str] = None

    @property
    def safe(self) -> bool:
        return not self.contains_entities


class SecretsDetectionResponse(SecretsDetectionSimpleResponse):
    detected_secrets: dict[str, list[str]]


class SecretsDetectionGuardrail(Guardrail):
    regex_model: RegexModel
    patterns: Union[dict[str, str], dict[str, list[str]]] = {}
    redaction: REDACTION

    def __init__(
        self,
        use_defaults: bool = True,
        redaction: REDACTION = REDACTION.REDACT_ALL,
        **kwargs,
    ):
        patterns = {}
        if use_defaults:
            patterns = DEFAULT_SECRETS_PATTERNS.copy()
        if kwargs.get("patterns"):
            patterns.update(kwargs["patterns"])

        # Create the RegexModel instance
        regex_model = RegexModel(patterns=patterns)

        # Initialize the base class with both the regex_model and patterns
        super().__init__(
            regex_model=regex_model,
            patterns=patterns,
            redaction=redaction,
        )

    @weave.op()
    def guard(
        self,
        prompt: str,
        return_detected_types: bool = True,
        **kwargs,
    ) -> SecretsDetectionResponse | SecretsDetectionResponse:
        """
        Check if the input prompt contains any entities based on the regex patterns.

        Args:
            prompt: Input text to check for entities
            return_detected_types: If True, returns detailed entity type information

        Returns:
            SecretsDetectionResponse or SecretsDetectionResponse containing detection results
        """
        result = self.regex_model.check(prompt)

        # Create detailed explanation
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

        if return_detected_types:
            return SecretsDetectionResponse(
                contains_secrets=not result.passed,
                detected_secrets=result.matched_patterns,
                explanation="\n".join(explanation_parts),
                redacted_text=redacted_text,
            )
        else:
            return SecretsDetectionSimpleResponse(
                contains_entities=not result.passed,
                explanation="\n".join(explanation_parts),
                redacted_text=redacted_text,
            )


def main():
    weave.init(project_name="parambharat/guardrails-genie")

    guardrail = SecretsDetectionGuardrail(redaction=REDACTION.REDACT_ALL)
    dataset = [
        {
            "input": 'I need to pass a key\naws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"',
        },
        {
            "input": "My github token is: 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",
        },
        {
            "input": "My JWT token is: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        },
    ]

    for item in dataset:
        # Check text for entities
        result = guardrail.guard(prompt=item["input"])

        # Access results
        print(f"Contains entities: {result.contains_secrets}")
        print(f"Detected entities: {result.detected_secrets}")
        print(f"Explanation: {result.explanation}")
        print(f"Anonymized text: {result.redacted_text}")
    # import regex as re
    #
    # sample_input = "My JWT token is: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    # jwt_pattern = DEFAULT_SECRETS_PATTERNS["JwtToken"][0]
    # print(jwt_pattern)
    # pattern = re.compile(jwt_pattern)
    # print(pattern)
    # print(pattern.findall(sample_input))

    # import pandas as pd
    #
    # df = pd.read_json("secrets_patterns_bak.jsonl", lines=True)
    # df.loc[:, "patterns"] = df["patterns"].map(lambda x: [i[2:-1] for i in x])
    # df.to_json("secrets_patterns.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
