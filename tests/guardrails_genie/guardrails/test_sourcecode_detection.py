import pytest

from guardrails_genie.guardrails.sourcecode_detection import SourceCodeDetector


@pytest.fixture
def mock_sourcecode_guard(monkeypatch):
    def _mock_guard(*args, **kwargs):
        prompt = kwargs.get("prompt")

        if "source code" in prompt:
            return {"safe": False, "summary": "The output contains source code."}
        else:
            return {"safe": True, "summary": "No source code detected in the text."}

    monkeypatch.setattr(
        "guardrails_genie.guardrails.sourcecode_detection.SourceCodeDetector.guard",
        _mock_guard,
    )


def test_sourcecode_guard_has_code(mock_sourcecode_guard):
    sourcecode_detector = SourceCodeDetector()
    response = sourcecode_detector.guard("This is a test with source code")
    assert response == {"safe": False, "summary": "The output contains source code."}


def test_soucecode_guard_no_code(mock_sourcecode_guard):
    sourcecode_detector = SourceCodeDetector()
    response = sourcecode_detector.guard("This is a test without any code")
    assert response == {"safe": True, "summary": "No source code detected in the text."}
