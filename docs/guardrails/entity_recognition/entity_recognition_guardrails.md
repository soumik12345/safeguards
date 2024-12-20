# Entity Recognition Guardrails

A collection of guardrails for detecting and anonymizing various types of entities in text, including PII (Personally Identifiable Information), restricted terms, and custom entities.

## Available Guardrails

### 1. Regex Entity Recognition
Simple pattern-based entity detection using regular expressions.

```python
from safeguards.guardrails.entity_recognition import RegexEntityRecognitionGuardrail

# Initialize with default PII patterns
guardrail = RegexEntityRecognitionGuardrail(should_anonymize=True)

# Or with custom patterns
custom_patterns = {
    "employee_id": r"EMP\d{6}",
    "project_code": r"PRJ-[A-Z]{2}-\d{4}"
}
guardrail = RegexEntityRecognitionGuardrail(patterns=custom_patterns, should_anonymize=True)
```

### 2. Presidio Entity Recognition
Advanced entity detection using Microsoft's Presidio analyzer.

```python
from safeguards.guardrails.entity_recognition import PresidioEntityRecognitionGuardrail

# Initialize with default entities
guardrail = PresidioEntityRecognitionGuardrail(should_anonymize=True)

# Or with specific entities
selected_entities = ["CREDIT_CARD", "US_SSN", "EMAIL_ADDRESS"]
guardrail = PresidioEntityRecognitionGuardrail(
    selected_entities=selected_entities,
    should_anonymize=True
)
```

### 3. Transformers Entity Recognition
Entity detection using transformer-based models.

```python
from safeguards.guardrails.entity_recognition import TransformersEntityRecognitionGuardrail

# Initialize with default model
guardrail = TransformersEntityRecognitionGuardrail(should_anonymize=True)

# Or with specific model and entities
guardrail = TransformersEntityRecognitionGuardrail(
    model_name="iiiorg/piiranha-v1-detect-personal-information",
    selected_entities=["GIVENNAME", "SURNAME", "EMAIL"],
    should_anonymize=True
)
```

### 4. LLM Judge for Restricted Terms
Advanced detection of restricted terms, competitor mentions, and brand protection using LLMs.

```python
from safeguards.guardrails.entity_recognition import RestrictedTermsJudge

# Initialize with OpenAI model
guardrail = RestrictedTermsJudge(should_anonymize=True)

# Check for specific terms
result = guardrail.guard(
    text="Let's implement features like Salesforce",
    custom_terms=["Salesforce", "Oracle", "AWS"]
)
```

## Usage

All guardrails follow a consistent interface:

```python
# Initialize a guardrail
guardrail = RegexEntityRecognitionGuardrail(should_anonymize=True)

# Check text for entities
result = guardrail.guard("Hello, my email is john@example.com")

# Access results
print(f"Contains entities: {result.contains_entities}")
print(f"Detected entities: {result.detected_entities}")
print(f"Explanation: {result.explanation}")
print(f"Anonymized text: {result.anonymized_text}")
```

## Evaluation Tools

The module includes comprehensive evaluation tools and test cases:

- `pii_examples/`: Test cases for PII detection
- `banned_terms_examples/`: Test cases for restricted terms
- Benchmark scripts for evaluating model performance

### Running Evaluations

```python
# PII Detection Benchmark
from safeguards.guardrails.entity_recognition.pii_examples.pii_benchmark import main
main()

# (TODO): Restricted Terms Testing
from safeguards.guardrails.entity_recognition.banned_terms_examples.banned_term_benchmark import main
main()
```

## Features

- Entity detection and anonymization
- Support for multiple detection methods (regex, Presidio, transformers, LLMs)
- Customizable entity types and patterns
- Detailed explanations of detected entities
- Comprehensive evaluation framework
- Support for custom terms and patterns
- Batch processing capabilities
- Performance metrics and benchmarking

## Response Format

All guardrails return responses with the following structure:

```python
{
    "contains_entities": bool,
    "detected_entities": {
        "entity_type": ["detected_value_1", "detected_value_2"]
    },
    "explanation": str,
    "anonymized_text": Optional[str]
}
```