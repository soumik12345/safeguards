from typing import Dict, Optional, ClassVar, List

import weave
from pydantic import BaseModel

from ...regex_model import RegexModel
from ..base import Guardrail
import re


class RegexEntityRecognitionResponse(BaseModel):
    contains_entities: bool
    detected_entities: Dict[str, list[str]]
    explanation: str
    anonymized_text: Optional[str] = None

    @property
    def safe(self) -> bool:
        return not self.contains_entities


class RegexEntityRecognitionSimpleResponse(BaseModel):
    contains_entities: bool
    explanation: str
    anonymized_text: Optional[str] = None

    @property
    def safe(self) -> bool:
        return not self.contains_entities


class RegexEntityRecognitionGuardrail(Guardrail):
    regex_model: RegexModel
    patterns: Dict[str, str] = {}
    should_anonymize: bool = False
    
    DEFAULT_PATTERNS: ClassVar[Dict[str, str]] = {
        "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "TELEPHONENUM": r'\b(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
        "SOCIALNUM": r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
        "CREDITCARDNUMBER": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        "DATEOFBIRTH": r'\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19|20)\d{2}\b',
        "DRIVERLICENSENUM": r'[A-Z]\d{7}',  # Example pattern, adjust for your needs
        "ACCOUNTNUM": r'\b\d{10,12}\b',  # Example pattern for bank accounts
        "ZIPCODE": r'\b\d{5}(?:-\d{4})?\b',
        "GIVENNAME": r'\b[A-Z][a-z]+\b',  # Basic pattern for first names
        "SURNAME": r'\b[A-Z][a-z]+\b',    # Basic pattern for last names
        "CITY": r'\b[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*\b',
        "STREET": r'\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b',
        "IDCARDNUM": r'[A-Z]\d{7,8}',  # Generic pattern for ID cards
        "USERNAME": r'@[A-Za-z]\w{3,}',  # Basic username pattern
        "PASSWORD": r'[A-Za-z0-9@#$%^&+=]{8,}',  # Basic password pattern
        "TAXNUM": r'\b\d{2}[-]\d{7}\b',  # Example tax number pattern
        "BUILDINGNUM": r'\b\d+[A-Za-z]?\b'  # Basic building number pattern
    }
    
    def __init__(self, use_defaults: bool = True, should_anonymize: bool = False, show_available_entities: bool = False, **kwargs):
        patterns = {}
        if use_defaults:
            patterns = self.DEFAULT_PATTERNS.copy()
        if kwargs.get("patterns"):
            patterns.update(kwargs["patterns"])

        if show_available_entities:
            self._print_available_entities(patterns.keys())
        
        # Create the RegexModel instance
        regex_model = RegexModel(patterns=patterns)
        
        # Initialize the base class with both the regex_model and patterns
        super().__init__(
            regex_model=regex_model, 
            patterns=patterns,
            should_anonymize=should_anonymize
        )

    def text_to_pattern(self, text: str) -> str:
        """
        Convert input text into a regex pattern that matches the exact text.
        """
        # Escape special regex characters in the text
        escaped_text = re.escape(text)
        # Create a pattern that matches the exact text, case-insensitive
        return rf"\b{escaped_text}\b"
    
    def _print_available_entities(self, entities: List[str]):
        """Print available entities"""
        print("\nAvailable entity types:")
        print("=" * 25)
        for entity in entities:
            print(f"- {entity}")
        print("=" * 25 + "\n")

    @weave.op()
    def guard(self, prompt: str, custom_terms: Optional[list[str]] = None, return_detected_types: bool = True, aggregate_redaction: bool = True, **kwargs) -> RegexEntityRecognitionResponse | RegexEntityRecognitionSimpleResponse:
        """
        Check if the input prompt contains any entities based on the regex patterns.
        
        Args:
            prompt: Input text to check for entities
            custom_terms: List of custom terms to be converted into regex patterns. If provided, 
                        only these terms will be checked, ignoring default patterns.
            return_detected_types: If True, returns detailed entity type information
            
        Returns:
            RegexEntityRecognitionResponse or RegexEntityRecognitionSimpleResponse containing detection results
        """
        if custom_terms:
            # Create a temporary RegexModel with only the custom patterns
            temp_patterns = {term: self.text_to_pattern(term) for term in custom_terms}
            temp_model = RegexModel(patterns=temp_patterns)
            result = temp_model.check(prompt)
        else:
            # Use the original regex_model if no custom terms provided
            result = self.regex_model.check(prompt)
        
        # Create detailed explanation
        explanation_parts = []
        if result.matched_patterns:
            explanation_parts.append("Found the following entities in the text:")
            for entity_type, matches in result.matched_patterns.items():
                explanation_parts.append(f"- {entity_type}: {len(matches)} instance(s)")
        else:
            explanation_parts.append("No entities detected in the text.")
        
        if result.failed_patterns:
            explanation_parts.append("\nChecked but did not find these entity types:")
            for pattern in result.failed_patterns:
                explanation_parts.append(f"- {pattern}")
                
        # Updated anonymization logic
        anonymized_text = None
        if getattr(self, 'should_anonymize', False) and result.matched_patterns:
            anonymized_text = prompt
            for entity_type, matches in result.matched_patterns.items():
                for match in matches:
                    replacement = "[redacted]" if aggregate_redaction else f"[{entity_type.upper()}]"
                    anonymized_text = anonymized_text.replace(match, replacement)
        
        if return_detected_types:
            return RegexEntityRecognitionResponse(
                contains_entities=not result.passed,
                detected_entities=result.matched_patterns,
                explanation="\n".join(explanation_parts),
                anonymized_text=anonymized_text
            )
        else:
            return RegexEntityRecognitionSimpleResponse(
                contains_entities=not result.passed,
                explanation="\n".join(explanation_parts),
                anonymized_text=anonymized_text
            )

    @weave.op()
    def predict(self, prompt: str, return_detected_types: bool = True, aggregate_redaction: bool = True, **kwargs) -> RegexEntityRecognitionResponse | RegexEntityRecognitionSimpleResponse:
        return self.guard(prompt, return_detected_types=return_detected_types, aggregate_redaction=aggregate_redaction, **kwargs)