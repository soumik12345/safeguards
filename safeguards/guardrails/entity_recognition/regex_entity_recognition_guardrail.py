import re
from typing import ClassVar, Dict, List, Optional

import weave
from pydantic import BaseModel

from safeguards.guardrails.base import Guardrail
from safeguards.regex_model import RegexModel


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
    """
    A guardrail class for recognizing and optionally anonymizing entities in text using regular expressions.

    This class extends the Guardrail base class and utilizes a RegexModel to detect entities in the input text
    based on predefined or custom regex patterns. It provides functionality to check for entities, anonymize
    detected entities, and return detailed information about the detected entities.

    !!! example "Using RegexEntityRecognitionGuardrail"
        ```python
        from guardrails_genie.guardrails.entity_recognition import RegexEntityRecognitionGuardrail

        # Initialize with default PII patterns
        guardrail = RegexEntityRecognitionGuardrail(should_anonymize=True)

        # Or with custom patterns
        custom_patterns = {
            "employee_id": r"EMP\d{6}",
            "project_code": r"PRJ-[A-Z]{2}-\d{4}"
        }
        guardrail = RegexEntityRecognitionGuardrail(patterns=custom_patterns, should_anonymize=True)
        ```

    Attributes:
        regex_model (RegexModel): An instance of RegexModel used for entity recognition.
        patterns (Dict[str, str]): A dictionary of regex patterns for entity recognition.
        should_anonymize (bool): A flag indicating whether detected entities should be anonymized.
        DEFAULT_PATTERNS (ClassVar[Dict[str, str]]): A dictionary of default regex patterns for common entities.

    Args:
        use_defaults (bool): If True, use default patterns. If False, use custom patterns.
        should_anonymize (bool): If True, anonymize detected entities.
        show_available_entities (bool): If True, print available entity types.
    """

    regex_model: RegexModel
    patterns: Dict[str, str] = {}
    should_anonymize: bool = False

    DEFAULT_PATTERNS: ClassVar[Dict[str, str]] = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "TELEPHONENUM": r"\b(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b",
        "SOCIALNUM": r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
        "CREDITCARDNUMBER": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "DATEOFBIRTH": r"\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19|20)\d{2}\b",
        "DRIVERLICENSENUM": r"[A-Z]\d{7}",  # Example pattern, adjust for your needs
        "ACCOUNTNUM": r"\b\d{10,12}\b",  # Example pattern for bank accounts
        "ZIPCODE": r"\b\d{5}(?:-\d{4})?\b",
        "GIVENNAME": r"\b[A-Z][a-z]+\b",  # Basic pattern for first names
        "SURNAME": r"\b[A-Z][a-z]+\b",  # Basic pattern for last names
        "CITY": r"\b[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*\b",
        "STREET": r"\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b",
        "IDCARDNUM": r"[A-Z]\d{7,8}",  # Generic pattern for ID cards
        "USERNAME": r"@[A-Za-z]\w{3,}",  # Basic username pattern
        "PASSWORD": r"[A-Za-z0-9@#$%^&+=]{8,}",  # Basic password pattern
        "TAXNUM": r"\b\d{2}[-]\d{7}\b",  # Example tax number pattern
        "BUILDINGNUM": r"\b\d+[A-Za-z]?\b",  # Basic building number pattern
    }

    def __init__(
        self,
        use_defaults: bool = True,
        should_anonymize: bool = False,
        show_available_entities: bool = False,
        **kwargs,
    ):
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
            should_anonymize=should_anonymize,
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
    def guard(
        self,
        prompt: str,
        custom_terms: Optional[list[str]] = None,
        return_detected_types: bool = True,
        aggregate_redaction: bool = True,
        **kwargs,
    ) -> RegexEntityRecognitionResponse | RegexEntityRecognitionSimpleResponse:
        """
        Analyzes the input prompt to detect entities based on predefined or custom regex patterns.

        This function checks the provided text (prompt) for entities using regex patterns. It can
        utilize either default patterns or custom terms provided by the user. If custom terms are
        specified, they are converted into regex patterns, and only these are used for entity detection.
        The function returns detailed information about detected entities and can optionally anonymize
        the detected entities in the text.

        Args:
            prompt (str): The input text to be analyzed for entity detection.
            custom_terms (Optional[list[str]]): A list of custom terms to be converted into regex patterns.
                If provided, only these terms will be checked, ignoring default patterns.
            return_detected_types (bool): If True, the function returns detailed information about the
                types of entities detected in the text.
            aggregate_redaction (bool): Determines the anonymization strategy. If True, all detected
                entities are replaced with a generic "[redacted]" label. If False, each entity type is
                replaced with its specific label (e.g., "[ENTITY_TYPE]").

        Returns:
            RegexEntityRecognitionResponse or RegexEntityRecognitionSimpleResponse: An object containing
            the results of the entity detection, including whether entities were found, the types and
            counts of detected entities, an explanation of the detection process, and optionally, the
            anonymized text.
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
        if getattr(self, "should_anonymize", False) and result.matched_patterns:
            anonymized_text = prompt
            for entity_type, matches in result.matched_patterns.items():
                for match in matches:
                    replacement = (
                        "[redacted]"
                        if aggregate_redaction
                        else f"[{entity_type.upper()}]"
                    )
                    anonymized_text = anonymized_text.replace(match, replacement)

        if return_detected_types:
            return RegexEntityRecognitionResponse(
                contains_entities=not result.passed,
                detected_entities=result.matched_patterns,
                explanation="\n".join(explanation_parts),
                anonymized_text=anonymized_text,
            )
        else:
            return RegexEntityRecognitionSimpleResponse(
                contains_entities=not result.passed,
                explanation="\n".join(explanation_parts),
                anonymized_text=anonymized_text,
            )

    @weave.op()
    def predict(
        self,
        prompt: str,
        return_detected_types: bool = True,
        aggregate_redaction: bool = True,
        **kwargs,
    ) -> RegexEntityRecognitionResponse | RegexEntityRecognitionSimpleResponse:
        return self.guard(
            prompt,
            return_detected_types=return_detected_types,
            aggregate_redaction=aggregate_redaction,
            **kwargs,
        )
