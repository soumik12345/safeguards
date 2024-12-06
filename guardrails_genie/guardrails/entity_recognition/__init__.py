import warnings

from .llm_judge_entity_recognition_guardrail import RestrictedTermsJudge

try:
    from .presidio_entity_recognition_guardrail import (
        PresidioEntityRecognitionGuardrail,
    )
except ImportError:
    warnings.warn(
        "Presidio is not installed. If you want to use `PresidioEntityRecognitionGuardrail`, you can install the required packages using `pip install -e .[presidio]`"
    )

from .regex_entity_recognition_guardrail import RegexEntityRecognitionGuardrail
from .transformers_entity_recognition_guardrail import (
    TransformersEntityRecognitionGuardrail,
)

__all__ = [
    "PresidioEntityRecognitionGuardrail",
    "RegexEntityRecognitionGuardrail",
    "TransformersEntityRecognitionGuardrail",
    "RestrictedTermsJudge",
]
