import hashlib
import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from safeguards.guardrails import SecretsDetectionGuardrail
from safeguards.guardrails.secrets_detection import (
    REDACTION,
    SecretsDetectionResponse,
    SecretsDetectionSimpleResponse,
    redact_value,
)


@pytest.fixture
def mock_secrets_guard(monkeypatch):
    def _mock_guard(*args, **kwargs):
        prompt = kwargs.get("prompt")
        return_detected_types = kwargs.get("return_detected_secrets")

        if "safe text" in prompt:
            if return_detected_types:
                return SecretsDetectionResponse(
                    contains_secrets=False,
                    explanation="No secrets detected in the text.",
                    detected_secrets={},
                    redacted_text=prompt,
                    risk_score=0.0,
                )
            else:
                return SecretsDetectionSimpleResponse(
                    contains_secrets=False,
                    explanation="No secrets detected in the text.",
                    redacted_text=prompt,
                    risk_score=0.0,
                )
        else:
            if return_detected_types:
                return SecretsDetectionResponse(
                    contains_secrets=True,
                    explanation="The output contains secrets.",
                    detected_secrets={"secrets": ["API_KEY"]},
                    redacted_text="My secret key is [REDACTED:]************[:REDACTED]",
                    risk_score=1.0,
                )
            else:
                return SecretsDetectionSimpleResponse(
                    contains_secrets=True,
                    explanation="The output contains secrets.",
                    redacted_text="My secret key is [REDACTED:]************[:REDACTED]",
                    risk_score=1.0,
                )

    monkeypatch.setattr(
        "guardrails_genie.guardrails.secrets_detection.SecretsDetectionGuardrail.guard",
        _mock_guard,
    )


def test_redact_partial():
    text = "ABCDEFGHIJKL"
    redacted_text = redact_value(text, REDACTION.REDACT_PARTIAL)
    assert redacted_text == "[REDACTED:]AB..KL[:REDACTED]"


def test_redact_all():
    text = "ABCDEFGHIJKL"
    redacted_text = redact_value(text, REDACTION.REDACT_ALL)
    assert redacted_text == "[REDACTED:]************[:REDACTED]"


def test_redact_hash():
    text = "ABCDEFGHIJKL"
    hashed_value = hashlib.md5(text.encode()).hexdigest()
    redacted_text = redact_value(text, REDACTION.REDACT_HASH)
    assert redacted_text == f"[REDACTED:]{hashed_value}[:REDACTED]"


def test_secrets_detection_guardrail_detect_types(mock_secrets_guard):
    from safeguards.guardrails.secrets_detection import (
        REDACTION,
        SecretsDetectionGuardrail,
    )

    guardrail = SecretsDetectionGuardrail(redaction=REDACTION.REDACT_ALL)
    prompt = "My secret key is ABCDEFGHIJKL"

    result = guardrail.guard(prompt=prompt, return_detected_secrets=True)

    assert result.contains_secrets is True
    assert result.explanation == "The output contains secrets."
    assert result.detected_secrets == {"secrets": ["API_KEY"]}
    assert result.redacted_text == "My secret key is [REDACTED:]************[:REDACTED]"


def test_secrets_detection_guardrail_simple_response(mock_secrets_guard):
    from safeguards.guardrails.secrets_detection import (
        REDACTION,
        SecretsDetectionGuardrail,
    )

    guardrail = SecretsDetectionGuardrail(redaction=REDACTION.REDACT_ALL)
    prompt = "My secret key is ABCDEFGHIJKL"

    result = guardrail.guard(prompt=prompt, return_detected_secrets=False)

    assert result.contains_secrets is True
    assert result.explanation == "The output contains secrets."
    assert result.redacted_text == "My secret key is [REDACTED:]************[:REDACTED]"


def test_secrets_detection_guardrail_no_secrets(mock_secrets_guard):
    from safeguards.guardrails.secrets_detection import (
        REDACTION,
        SecretsDetectionGuardrail,
    )

    guardrail = SecretsDetectionGuardrail(redaction=REDACTION.REDACT_ALL)
    prompt = "This is a safe text with no secrets."

    result = guardrail.guard(prompt=prompt, return_detected_secrets=True)

    assert result.contains_secrets is False
    assert result.explanation == "No secrets detected in the text."
    assert result.detected_secrets == {}
    assert result.redacted_text == prompt


def pattern_strategy(pattern):
    return st.from_regex(re.compile(pattern), fullmatch=True)


@settings(deadline=1000)
@given(pattern_strategy(r"AKIA[0-9A-Z]{16}"))
def test_specific_pattern_guardrail(text):
    guardrail = SecretsDetectionGuardrail(redaction=REDACTION.REDACT_ALL)
    result = guardrail.guard(prompt=text, return_detected_secrets=True)

    assert result.contains_secrets is True
    assert "AWS Access Key" in result.detected_secrets
