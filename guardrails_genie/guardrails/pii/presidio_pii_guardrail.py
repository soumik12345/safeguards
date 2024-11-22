from typing import List, Dict, Optional, ClassVar
import weave
from pydantic import BaseModel

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from ..base import Guardrail

class PresidioPIIGuardrailResponse(BaseModel):
    contains_pii: bool
    detected_pii_types: Dict[str, List[str]]
    safe_to_process: bool
    explanation: str
    anonymized_text: Optional[str] = None

#TODO: Add support for transformers workflow and not just Spacy
class PresidioPIIGuardrail(Guardrail):
    AVAILABLE_ENTITIES: ClassVar[List[str]] = [
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", 
        "CREDIT_CARD", "CRYPTO", "DATE_TIME", "NRP", "MEDICAL_LICENSE",
        "URL", "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_ITIN", 
        "US_PASSPORT", "US_SSN", "UK_NHS", "IP_ADDRESS"
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
        language: str = "en"
    ):
        # Initialize default values
        if selected_entities is None:
            selected_entities = [
                "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", 
                "LOCATION", "CREDIT_CARD", "US_SSN"
            ]
        
        # Validate selected entities
        invalid_entities = set(selected_entities) - set(self.AVAILABLE_ENTITIES)
        if invalid_entities:
            raise ValueError(f"Invalid entities: {invalid_entities}")
            
        # Initialize Presidio engines
        analyzer = AnalyzerEngine()
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
    def guard(self, prompt: str, **kwargs) -> PresidioPIIGuardrailResponse:
        """
        Check if the input prompt contains any PII using Presidio.
        """
        # Analyze text for PII
        analyzer_results = self.analyzer.analyze(
            text=prompt,
            entities=self.selected_entities,
            language=self.language
        )
        
        # Group results by entity type
        detected_pii = {}
        for result in analyzer_results:
            entity_type = result.entity_type
            text_slice = prompt[result.start:result.end]
            if entity_type not in detected_pii:
                detected_pii[entity_type] = []
            detected_pii[entity_type].append(text_slice)
        
        # Create explanation
        explanation_parts = []
        if detected_pii:
            explanation_parts.append("Found the following PII in the text:")
            for pii_type, instances in detected_pii.items():
                explanation_parts.append(f"- {pii_type}: {len(instances)} instance(s)")
        else:
            explanation_parts.append("No PII detected in the text.")
            
        # Add information about what was checked
        explanation_parts.append("\nChecked for these PII types:")
        for entity in self.selected_entities:
            explanation_parts.append(f"- {entity}")
        
        # Anonymize if requested
        anonymized_text = None
        if self.should_anonymize and detected_pii:
            anonymized_result = self.anonymizer.anonymize(
                text=prompt,
                analyzer_results=analyzer_results
            )
            anonymized_text = anonymized_result.text
            
        return PresidioPIIGuardrailResponse(
            contains_pii=bool(detected_pii),
            detected_pii_types=detected_pii,
            safe_to_process=not bool(detected_pii),
            explanation="\n".join(explanation_parts),
            anonymized_text=anonymized_text
        )
