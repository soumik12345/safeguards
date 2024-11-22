from typing import Dict, Optional, ClassVar

import weave
from pydantic import BaseModel

from ...regex_model import RegexModel
from ..base import Guardrail


class RegexPIIGuardrailResponse(BaseModel):
    contains_pii: bool
    detected_pii_types: Dict[str, list[str]]
    safe_to_process: bool
    explanation: str


class RegexPIIGuardrail(Guardrail):
    regex_model: RegexModel
    patterns: Dict[str, str] = {}
    
    DEFAULT_PII_PATTERNS: ClassVar[Dict[str, str]] = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone_number": r"\b(?:\+?1[-.]?)?\(?(?:[0-9]{3})\)?[-.]?(?:[0-9]{3})[-.]?(?:[0-9]{4})\b",
        "ssn": r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
        "credit_card": r"\b\d{4}[-.]?\d{4}[-.]?\d{4}[-.]?\d{4}\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        "date_of_birth": r"\b\d{2}[-/]\d{2}[-/]\d{4}\b",
        "passport": r"\b[A-Z]{1,2}[0-9]{6,9}\b",
        "drivers_license": r"\b[A-Z]\d{7}\b",
        "bank_account": r"\b\d{8,17}\b",
        "zip_code": r"\b\d{5}(?:[-]\d{4})?\b"
    }
    
    def __init__(self, use_defaults: bool = True, **kwargs):
        patterns = {}
        if use_defaults:
            patterns = self.DEFAULT_PII_PATTERNS.copy()
        if kwargs.get("patterns"):
            patterns.update(kwargs["patterns"])
        
        # Create the RegexModel instance
        regex_model = RegexModel(patterns=patterns)
        
        # Initialize the base class with both the regex_model and patterns
        super().__init__(regex_model=regex_model, patterns=patterns)

    @weave.op()
    def guard(self, prompt: str, **kwargs) -> RegexPIIGuardrailResponse:
        """
        Check if the input prompt contains any PII based on the regex patterns.
        
        Args:
            prompt: Input text to check for PII
            
        Returns:
            RegexPIIGuardrailResponse containing PII detection results and recommendations
        """
        result = self.regex_model.check(prompt)
        
        # Create detailed explanation
        explanation_parts = []
        if result.matched_patterns:
            explanation_parts.append("Found the following PII in the text:")
            for pii_type, matches in result.matched_patterns.items():
                explanation_parts.append(f"- {pii_type}: {len(matches)} instance(s)")
        else:
            explanation_parts.append("No PII detected in the text.")
        
        if result.failed_patterns:
            explanation_parts.append("\nChecked but did not find these PII types:")
            for pattern in result.failed_patterns:
                explanation_parts.append(f"- {pattern}")
                
        return RegexPIIGuardrailResponse(
            contains_pii=not result.passed,
            detected_pii_types=result.matched_patterns,
            safe_to_process=result.passed,
            explanation="\n".join(explanation_parts)
        )