from typing import List, Dict, Optional, ClassVar, Any
import weave
from pydantic import BaseModel

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine

from ..base import Guardrail

class PresidioEntityRecognitionResponse(BaseModel):
    contains_entities: bool
    detected_entities: Dict[str, List[str]]
    explanation: str
    anonymized_text: Optional[str] = None

class PresidioEntityRecognitionSimpleResponse(BaseModel):
    contains_entities: bool
    explanation: str
    anonymized_text: Optional[str] = None

#TODO: Add support for transformers workflow and not just Spacy
class PresidioEntityRecognitionGuardrail(Guardrail):
    @staticmethod
    def get_available_entities() -> List[str]:
        registry = RecognizerRegistry()
        analyzer = AnalyzerEngine(registry=registry)
        return [recognizer.supported_entities[0] 
                for recognizer in analyzer.registry.recognizers]
    
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
        custom_recognizers: Optional[List[Any]] = None
    ):
        # Initialize default values
        if selected_entities is None:
            selected_entities = [
                "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", 
                "LOCATION", "CREDIT_CARD", "US_SSN"
            ]
        
        # Get available entities dynamically
        available_entities = self.get_available_entities()
        
        # Validate selected entities
        invalid_entities = set(selected_entities) - set(available_entities)
        if invalid_entities:
            raise ValueError(f"Invalid entities: {invalid_entities}")
            
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
                    supported_entity=entity_type,
                    deny_list=tokens
                )
                analyzer.registry.add_recognizer(deny_list_recognizer)
        
        # Add regex pattern recognizers if provided
        if regex_patterns:
            for entity_type, patterns in regex_patterns.items():
                presidio_patterns = [
                    Pattern(
                        name=pattern.get("name", f"pattern_{i}"),
                        regex=pattern["regex"],
                        score=pattern.get("score", 0.5)
                    ) for i, pattern in enumerate(patterns)
                ]
                regex_recognizer = PatternRecognizer(
                    supported_entity=entity_type,
                    patterns=presidio_patterns
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
            language=language
        )

    @weave.op()
    def guard(self, prompt: str, return_detected_types: bool = True, **kwargs) -> PresidioEntityRecognitionResponse | PresidioEntityRecognitionSimpleResponse:
        """
        Check if the input prompt contains any entities using Presidio.
        
        Args:
            prompt: The text to analyze
            return_detected_types: If True, returns detailed entity type information
        """
        # Analyze text for entities
        analyzer_results = self.analyzer.analyze(
            text=prompt,
            entities=self.selected_entities,
            language=self.language
        )
        
        # Group results by entity type
        detected_entities = {}
        for result in analyzer_results:
            entity_type = result.entity_type
            text_slice = prompt[result.start:result.end]
            if entity_type not in detected_entities:
                detected_entities[entity_type] = []
            detected_entities[entity_type].append(text_slice)
        
        # Create explanation
        explanation_parts = []
        if detected_entities:
            explanation_parts.append("Found the following entities in the text:")
            for entity_type, instances in detected_entities.items():
                explanation_parts.append(f"- {entity_type}: {len(instances)} instance(s)")
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
                text=prompt,
                analyzer_results=analyzer_results
            )
            anonymized_text = anonymized_result.text
            
        if return_detected_types:
            return PresidioEntityRecognitionResponse(
                contains_entities=bool(detected_entities),
                detected_entities=detected_entities,
                explanation="\n".join(explanation_parts),
                anonymized_text=anonymized_text
            )
        else:
            return PresidioEntityRecognitionSimpleResponse(
                contains_entities=bool(detected_entities),
                explanation="\n".join(explanation_parts),
                anonymized_text=anonymized_text
            )
    
    @weave.op()
    def predict(self, prompt: str, return_detected_types: bool = True, **kwargs) -> PresidioEntityRecognitionResponse | PresidioEntityRecognitionSimpleResponse:
        return self.guard(prompt, return_detected_types=return_detected_types, **kwargs)