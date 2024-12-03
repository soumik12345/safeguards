import hashlib
import re

import pytest
from hypothesis import given, strategies as st, settings

from guardrails_genie.guardrails.secrets_detection import (
    DEFAULT_SECRETS_PATTERNS,
    SecretsDetectionSimpleResponse,
    SecretsDetectionResponse,
    SecretsDetectionGuardrail,
    REDACTION,
    redact,
)


@pytest.fixture
def mock_secrets_guard(monkeypatch):
    def _mock_guard(*args, **kwargs):
        prompt = kwargs.get("prompt")
        return_detected_types = kwargs.get("return_detected_types")

        if "safe text" in prompt:
            if return_detected_types:
                return SecretsDetectionResponse(
                    contains_secrets=False,
                    explanation="No secrets detected in the text.",
                    detected_secrets={},
                    redacted_text=prompt,
                )
            else:
                return SecretsDetectionSimpleResponse(
                    contains_secrets=False,
                    explanation="No secrets detected in the text.",
                    redacted_text=prompt,
                )
        else:
            if return_detected_types:
                return SecretsDetectionResponse(
                    contains_secrets=True,
                    explanation="The output contains secrets.",
                    detected_secrets={"secrets": ["API_KEY"]},
                    redacted_text="My secret key is [REDACTED:]************[:REDACTED]",
                )
            else:
                return SecretsDetectionSimpleResponse(
                    contains_secrets=True,
                    explanation="The output contains secrets.",
                    redacted_text="My secret key is [REDACTED:]************[:REDACTED]",
                )

    monkeypatch.setattr(
        "guardrails_genie.guardrails.secrets_detection.SecretsDetectionGuardrail.guard",
        _mock_guard,
    )


def test_redact_partial():
    text = "My secret key is ABCDEFGHIJKL"
    matches = ["ABCDEFGHIJKL"]
    redacted_text = redact(text, matches, REDACTION.REDACT_PARTIAL)
    assert redacted_text == "My secret key is [REDACTED:]AB..KL[:REDACTED]"


def test_redact_all():
    text = "My secret key is ABCDEFGHIJKL"
    matches = ["ABCDEFGHIJKL"]
    redacted_text = redact(text, matches, REDACTION.REDACT_ALL)
    assert redacted_text == "My secret key is [REDACTED:]************[:REDACTED]"


def test_redact_hash():
    text = "My secret key is ABCDEFGHIJKL"
    matches = ["ABCDEFGHIJKL"]
    hashed_value = hashlib.md5("ABCDEFGHIJKL".encode()).hexdigest()
    redacted_text = redact(text, matches, REDACTION.REDACT_HASH)
    assert redacted_text == f"My secret key is [REDACTED:]{hashed_value}[:REDACTED]"


def test_redact_no_match():
    text = "My secret key is ABCDEFGHIJKL"
    matches = ["XYZ"]
    redacted_text = redact(text, matches, REDACTION.REDACT_ALL)
    assert redacted_text == text


def test_secrets_detection_guardrail_detect_types(mock_secrets_guard):
    from guardrails_genie.guardrails.secrets_detection import (
        SecretsDetectionGuardrail,
        REDACTION,
    )

    guardrail = SecretsDetectionGuardrail(redaction=REDACTION.REDACT_ALL)
    prompt = "My secret key is ABCDEFGHIJKL"

    result = guardrail.guard(prompt=prompt, return_detected_types=True)

    assert result.contains_secrets is True
    assert result.explanation == "The output contains secrets."
    assert result.detected_secrets == {"secrets": ["API_KEY"]}
    assert result.redacted_text == "My secret key is [REDACTED:]************[:REDACTED]"


def test_secrets_detection_guardrail_simple_response(mock_secrets_guard):
    from guardrails_genie.guardrails.secrets_detection import (
        SecretsDetectionGuardrail,
        REDACTION,
    )

    guardrail = SecretsDetectionGuardrail(redaction=REDACTION.REDACT_ALL)
    prompt = "My secret key is ABCDEFGHIJKL"

    result = guardrail.guard(prompt=prompt, return_detected_types=False)

    assert result.contains_secrets is True
    assert result.explanation == "The output contains secrets."
    assert result.redacted_text == "My secret key is [REDACTED:]************[:REDACTED]"


def test_secrets_detection_guardrail_no_secrets(mock_secrets_guard):
    from guardrails_genie.guardrails.secrets_detection import (
        SecretsDetectionGuardrail,
        REDACTION,
    )

    guardrail = SecretsDetectionGuardrail(redaction=REDACTION.REDACT_ALL)
    prompt = "This is a safe text with no secrets."

    result = guardrail.guard(prompt=prompt, return_detected_types=True)

    assert result.contains_secrets is False
    assert result.explanation == "No secrets detected in the text."
    assert result.detected_secrets == {}
    assert result.redacted_text == prompt


# Create a strategy to generate strings that match the patterns
def pattern_strategy(pattern):
    return st.from_regex(re.compile(pattern), fullmatch=True)


@settings(deadline=1000)  # Set the deadline to 1000 milliseconds (1 second)
@given(pattern_strategy(DEFAULT_SECRETS_PATTERNS["JwtToken"][0]))
def test_specific_pattern_guardrail(text):
    guardrail = SecretsDetectionGuardrail(redaction=REDACTION.REDACT_ALL)
    result = guardrail.guard(prompt=text, return_detected_types=True)

    assert result.contains_secrets is True
    assert "JwtToken" in result.detected_secrets
