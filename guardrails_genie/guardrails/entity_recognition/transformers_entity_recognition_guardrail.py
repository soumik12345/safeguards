from typing import List, Dict, Optional, ClassVar
from transformers import pipeline, AutoConfig
import json
from pydantic import BaseModel
from ..base import Guardrail
import weave

class TransformersEntityRecognitionResponse(BaseModel):
    contains_entities: bool
    detected_entities: Dict[str, List[str]]
    explanation: str
    anonymized_text: Optional[str] = None

    @property
    def safe(self) -> bool:
        return not self.contains_entities

class TransformersEntityRecognitionSimpleResponse(BaseModel):
    contains_entities: bool
    explanation: str
    anonymized_text: Optional[str] = None

    @property
    def safe(self) -> bool:
        return not self.contains_entities

class TransformersEntityRecognitionGuardrail(Guardrail):
    """Generic guardrail for detecting entities using any token classification model."""
    
    _pipeline: Optional[object] = None
    selected_entities: List[str]
    should_anonymize: bool
    available_entities: List[str]
    
    def __init__(
        self,
        model_name: str = "iiiorg/piiranha-v1-detect-personal-information",
        selected_entities: Optional[List[str]] = None,
        should_anonymize: bool = False,
        show_available_entities: bool = False,
    ):
        # Load model config and extract available entities
        config = AutoConfig.from_pretrained(model_name)
        entities = self._extract_entities_from_config(config)
        
        if show_available_entities:
            self._print_available_entities(entities)
        
        # Initialize default values if needed
        if selected_entities is None:
            selected_entities = entities  # Use all available entities by default
            
        # Filter out invalid entities and warn user
        invalid_entities = [e for e in selected_entities if e not in entities]
        valid_entities = [e for e in selected_entities if e in entities]
        
        if invalid_entities:
            print(f"\nWarning: The following entities are not available and will be ignored: {invalid_entities}")
            print(f"Continuing with valid entities: {valid_entities}")
            selected_entities = valid_entities
        
        # Call parent class constructor
        super().__init__(
            selected_entities=selected_entities,
            should_anonymize=should_anonymize,
            available_entities=entities
        )
        
        # Initialize pipeline
        self._pipeline = pipeline(
            task="token-classification",
            model=model_name,
            aggregation_strategy="simple"  # Merge same entities
        )

    def _extract_entities_from_config(self, config) -> List[str]:
        """Extract unique entity types from the model config."""
        # Get id2label mapping from config
        id2label = config.id2label
        
        # Extract unique entity types (removing B- and I- prefixes)
        entities = set()
        for label in id2label.values():
            if label.startswith(('B-', 'I-')):
                entities.add(label[2:])  # Remove prefix
            elif label != 'O':  # Skip the 'O' (Outside) label
                entities.add(label)
        
        return sorted(list(entities))

    def _print_available_entities(self, entities: List[str]):
        """Print all available entity types that can be detected by the model."""
        print("\nAvailable entity types:")
        print("=" * 25)
        for entity in entities:
            print(f"- {entity}")
        print("=" * 25 + "\n")

    def print_available_entities(self):
        """Print all available entity types that can be detected by the model."""
        self._print_available_entities(self.available_entities)

    def _detect_entities(self, text: str) -> Dict[str, List[str]]:
        """Detect entities in the text using the pipeline."""
        results = self._pipeline(text)
        
        # Group findings by entity type
        detected_entities = {}
        for entity in results:
            entity_type = entity['entity_group']
            if entity_type in self.selected_entities:
                if entity_type not in detected_entities:
                    detected_entities[entity_type] = []
                detected_entities[entity_type].append(entity['word'])
                
        return detected_entities

    def _anonymize_text(self, text: str, aggregate_redaction: bool = True) -> str:
        """Anonymize detected entities in text using the pipeline."""
        results = self._pipeline(text)
        
        # Sort entities by start position in reverse order to avoid offset issues
        entities = sorted(results, key=lambda x: x['start'], reverse=True)
        
        # Create a mutable list of characters
        chars = list(text)
        
        # Apply redactions
        for entity in entities:
            if entity['entity_group'] in self.selected_entities:
                start, end = entity['start'], entity['end']
                replacement = ' [redacted] ' if aggregate_redaction else f" [{entity['entity_group']}] "
                
                # Replace the entity with the redaction marker
                chars[start:end] = replacement
        
        # Join characters and clean up only consecutive spaces (preserving newlines)
        result = ''.join(chars)
        # Replace multiple spaces with single space, but preserve newlines
        lines = result.split('\n')
        cleaned_lines = [' '.join(line.split()) for line in lines]
        return '\n'.join(cleaned_lines)

    @weave.op()
    def guard(self, prompt: str, return_detected_types: bool = True, aggregate_redaction: bool = True) -> TransformersEntityRecognitionResponse | TransformersEntityRecognitionSimpleResponse:
        """Check if the input prompt contains any entities using the transformer pipeline.
        
        Args:
            prompt: The text to analyze
            return_detected_types: If True, returns detailed entity type information
            aggregate_redaction: If True, uses generic [redacted] instead of entity type
        """
        # Detect entities
        detected_entities = self._detect_entities(prompt)
        
        # Create explanation
        explanation_parts = []
        if detected_entities:
            explanation_parts.append("Found the following entities in the text:")
            for entity_type, instances in detected_entities.items():
                explanation_parts.append(f"- {entity_type}: {len(instances)} instance(s)")
        else:
            explanation_parts.append("No entities detected in the text.")
        
        explanation_parts.append("\nChecked for these entities:")
        for entity in self.selected_entities:
            explanation_parts.append(f"- {entity}")
        
        # Anonymize if requested
        anonymized_text = None
        if self.should_anonymize and detected_entities:
            anonymized_text = self._anonymize_text(prompt, aggregate_redaction)
        
        if return_detected_types:
            return TransformersEntityRecognitionResponse(
                contains_entities=bool(detected_entities),
                detected_entities=detected_entities,
                explanation="\n".join(explanation_parts),
                anonymized_text=anonymized_text
            )
        else:
            return TransformersEntityRecognitionSimpleResponse(
                contains_entities=bool(detected_entities),
                explanation="\n".join(explanation_parts),
                anonymized_text=anonymized_text
            )

    @weave.op()
    def predict(self, prompt: str, return_detected_types: bool = True, aggregate_redaction: bool = True, **kwargs) -> TransformersEntityRecognitionResponse | TransformersEntityRecognitionSimpleResponse:
        return self.guard(prompt, return_detected_types=return_detected_types, aggregate_redaction=aggregate_redaction, **kwargs)
