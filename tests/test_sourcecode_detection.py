import pytest

from safeguards.guardrails import SourceCodeDetectionGuardrail


@pytest.fixture
def mock_sourcecode_guard(monkeypatch):
    def _mock_guard(*args, **kwargs):
        prompt = kwargs.get("prompt")

        if "source code" in prompt:
            return {"safe": False, "summary": "The output contains source code."}
        else:
            return {"safe": True, "summary": "No source code detected in the text."}

    monkeypatch.setattr(
        "safeguards.guardrails.sourcecode_detection.SourceCodeDetectionGuardrail.guard",
        _mock_guard,
    )


def test_sourcecode_guard_has_code(mock_sourcecode_guard):
    sourcecode_detector = SourceCodeDetectionGuardrail()
    response = sourcecode_detector.guard(prompt="This is a test with source code")
    assert response == {"safe": False, "summary": "The output contains source code."}


def test_soucecode_guard_no_code(mock_sourcecode_guard):
    sourcecode_detector = SourceCodeDetectionGuardrail()
    response = sourcecode_detector.guard(prompt="This is a test without any code")
    assert response == {"safe": True, "summary": "No source code detected in the text."}
