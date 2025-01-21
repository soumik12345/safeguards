from typing import Any, Dict, List, Optional

import weave
from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
    RecognizerRegistry,
)
from presidio_anonymizer import AnonymizerEngine
from pydantic import BaseModel

from safeguards.guardrails.base import Guardrail


class PresidioEntityRecognitionResponse(BaseModel):
    contains_entities: bool
    detected_entities: Dict[str, List[str]]
    explanation: str
    anonymized_text: Optional[str] = None

    @property
    def safe(self) -> bool:
        return not self.contains_entities


class PresidioEntityRecognitionSimpleResponse(BaseModel):
    contains_entities: bool
    explanation: str
    anonymized_text: Optional[str] = None

    @property
    def safe(self) -> bool:
        return not self.contains_entities


# TODO: Add support for transformers workflow and not just Spacy
class PresidioEntityRecognitionGuardrail(Guardrail):
    """
    A guardrail class for entity recognition and anonymization using Presidio.

    This class extends the Guardrail base class to provide functionality for
    detecting and optionally anonymizing entities in text using the Presidio
    library. It leverages Presidio's AnalyzerEngine and AnonymizerEngine to
    perform these tasks.

    !!! example "Using PresidioEntityRecognitionGuardrail"
        ```python
        from guardrails_genie.guardrails.entity_recognition import PresidioEntityRecognitionGuardrail

        # Initialize with default entities
        guardrail = PresidioEntityRecognitionGuardrail(should_anonymize=True)

        # Or with specific entities
        selected_entities = ["CREDIT_CARD", "US_SSN", "EMAIL_ADDRESS"]
        guardrail = PresidioEntityRecognitionGuardrail(
            selected_entities=selected_entities,
            should_anonymize=True
        )
        ```

    Attributes:
        analyzer (AnalyzerEngine): The Presidio engine used for entity analysis.
        anonymizer (AnonymizerEngine): The Presidio engine used for text anonymization.
        selected_entities (List[str]): A list of entity types to detect in the text.
        should_anonymize (bool): A flag indicating whether detected entities should be anonymized.
        language (str): The language of the text to be analyzed.

    Args:
        selected_entities (Optional[List[str]]): A list of entity types to detect in the text.
        should_anonymize (bool): A flag indicating whether detected entities should be anonymized.
        language (str): The language of the text to be analyzed.
        deny_lists (Optional[Dict[str, List[str]]]): A dictionary of entity types and their
            corresponding deny lists.
        regex_patterns (Optional[Dict[str, List[Dict[str, str]]]]): A dictionary of entity
            types and their corresponding regex patterns.
        custom_recognizers (Optional[List[Any]]): A list of custom recognizers to add to the
            analyzer.
        show_available_entities (bool): A flag indicating whether to print available entities.
    """

    @staticmethod
    def get_available_entities() -> List[str]:
        registry = RecognizerRegistry()
        analyzer = AnalyzerEngine(registry=registry)
        return [
            recognizer.supported_entities[0]
            for recognizer in analyzer.registry.recognizers
        ]

    analyzer: AnalyzerEngine
    anonymizer: AnonymizerEngine
    selected_entities: List[str]
    should_anonymize: bool
    language: str

    def __init__(
        self,
        selected_entities: Optional[List[str]] = None,
        should_anonymize: bool = False,
        language: str = "en",
        deny_lists: Optional[Dict[str, List[str]]] = None,
        regex_patterns: Optional[Dict[str, List[Dict[str, str]]]] = None,
        custom_recognizers: Optional[List[Any]] = None,
        show_available_entities: bool = False,
    ):
        # If show_available_entities is True, print available entities
        if show_available_entities:
            available_entities = self.get_available_entities()
            print("\nAvailable entities:")
            print("=" * 25)
            for entity in available_entities:
                print(f"- {entity}")
            print("=" * 25 + "\n")

        # Initialize default values to all available entities
        if selected_entities is None:
            selected_entities = self.get_available_entities()

        # Get available entities dynamically
        available_entities = self.get_available_entities()

        # Filter out invalid entities and warn user
        invalid_entities = [e for e in selected_entities if e not in available_entities]
        valid_entities = [e for e in selected_entities if e in available_entities]

        if invalid_entities:
            print(
                f"\nWarning: The following entities are not available and will be ignored: {invalid_entities}"
            )
            print(f"Continuing with valid entities: {valid_entities}")
            selected_entities = valid_entities

        # Initialize analyzer with default recognizers
        analyzer = AnalyzerEngine()

        # Add custom recognizers if provided
        if custom_recognizers:
            for recognizer in custom_recognizers:
                analyzer.registry.add_recognizer(recognizer)

        # Add deny list recognizers if provided
        if deny_lists:
            for entity_type, tokens in deny_lists.items():
                deny_list_recognizer = PatternRecognizer(
                    supported_entity=entity_type, deny_list=tokens
                )
                analyzer.registry.add_recognizer(deny_list_recognizer)

        # Add regex pattern recognizers if provided
        if regex_patterns:
            for entity_type, patterns in regex_patterns.items():
                presidio_patterns = [
                    Pattern(
                        name=pattern.get("name", f"pattern_{i}"),
                        regex=pattern["regex"],
                        score=pattern.get("score", 0.5),
                    )
                    for i, pattern in enumerate(patterns)
                ]
                regex_recognizer = PatternRecognizer(
                    supported_entity=entity_type, patterns=presidio_patterns
                )
                analyzer.registry.add_recognizer(regex_recognizer)

        # Initialize Presidio engines
        anonymizer = AnonymizerEngine()

        # Call parent class constructor with all fields
        super().__init__(
            analyzer=analyzer,
            anonymizer=anonymizer,
            selected_entities=selected_entities,
            should_anonymize=should_anonymize,
            language=language,
        )

    @weave.op()
    def guard(
        self, prompt: str, return_detected_types: bool = True, **kwargs
    ) -> PresidioEntityRecognitionResponse | PresidioEntityRecognitionSimpleResponse:
        """
        Analyzes the input prompt for entity recognition using the Presidio framework.

        This function utilizes the Presidio AnalyzerEngine to detect entities within the
        provided text prompt. It supports custom recognizers, deny lists, and regex patterns
        for entity detection. The detected entities are grouped by their types and an
        explanation of the findings is generated. If anonymization is enabled, the detected
        entities in the text are anonymized.

        Args:
            prompt (str): The text to be analyzed for entity recognition.
            return_detected_types (bool): Determines the type of response. If True, the
                response includes detailed information about detected entity types.

        Returns:
            PresidioEntityRecognitionResponse | PresidioEntityRecognitionSimpleResponse:
            A response object containing information about whether entities were detected,
            the types and instances of detected entities, an explanation of the analysis,
            and optionally, the anonymized text if anonymization is enabled.
        """
        # Analyze text for entities
        analyzer_results = self.analyzer.analyze(
            text=str(prompt), entities=self.selected_entities, language=self.language
        )

        # Group results by entity type
        detected_entities = {}
        for result in analyzer_results:
            entity_type = result.entity_type
            text_slice = prompt[result.start : result.end]
            if entity_type not in detected_entities:
                detected_entities[entity_type] = []
            detected_entities[entity_type].append(text_slice)

        # Create explanation
        explanation_parts = []
        if detected_entities:
            explanation_parts.append("Found the following entities in the text:")
            for entity_type, instances in detected_entities.items():
                explanation_parts.append(
                    f"- {entity_type}: {len(instances)} instance(s)"
                )
        else:
            explanation_parts.append("No entities detected in the text.")

        # Add information about what was checked
        explanation_parts.append("\nChecked for these entity types:")
        for entity in self.selected_entities:
            explanation_parts.append(f"- {entity}")

        # Anonymize if requested
        anonymized_text = None
        if self.should_anonymize and detected_entities:
            anonymized_result = self.anonymizer.anonymize(
                text=prompt, analyzer_results=analyzer_results
            )
            anonymized_text = anonymized_result.text

        if return_detected_types:
            return PresidioEntityRecognitionResponse(
                contains_entities=bool(detected_entities),
                detected_entities=detected_entities,
                explanation="\n".join(explanation_parts),
                anonymized_text=anonymized_text,
            )
        else:
            return PresidioEntityRecognitionSimpleResponse(
                contains_entities=bool(detected_entities),
                explanation="\n".join(explanation_parts),
                anonymized_text=anonymized_text,
            )

    @weave.op()
    def predict(
        self, prompt: str, return_detected_types: bool = True, **kwargs
    ) -> PresidioEntityRecognitionResponse | PresidioEntityRecognitionSimpleResponse:
        return self.guard(prompt, return_detected_types=return_detected_types, **kwargs)
