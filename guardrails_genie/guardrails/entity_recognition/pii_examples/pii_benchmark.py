from datasets import load_dataset
from typing import Dict, List, Tuple
import random
from tqdm import tqdm
import json
from pathlib import Path
import weave

# Add this mapping dictionary near the top of the file
PRESIDIO_TO_TRANSFORMER_MAPPING = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "TELEPHONENUM",
    "US_SSN": "SOCIALNUM",
    "CREDIT_CARD": "CREDITCARDNUMBER",
    "IP_ADDRESS": "IDCARDNUM",
    "DATE_TIME": "DATEOFBIRTH",
    "US_PASSPORT": "IDCARDNUM",
    "US_DRIVER_LICENSE": "DRIVERLICENSENUM",
    "US_BANK_NUMBER": "ACCOUNTNUM",
    "LOCATION": "CITY",
    "URL": "USERNAME",  # URLs often contain usernames
    "IN_PAN": "TAXNUM",  # Indian Permanent Account Number
    "UK_NHS": "IDCARDNUM",
    "SG_NRIC_FIN": "IDCARDNUM",
    "AU_ABN": "TAXNUM",  # Australian Business Number
    "AU_ACN": "TAXNUM",  # Australian Company Number
    "AU_TFN": "TAXNUM",  # Australian Tax File Number
    "AU_MEDICARE": "IDCARDNUM",
    "IN_AADHAAR": "IDCARDNUM",  # Indian national ID
    "IN_VOTER": "IDCARDNUM",
    "IN_PASSPORT": "IDCARDNUM",
    "CRYPTO": "ACCOUNTNUM",  # Cryptocurrency addresses
    "IBAN_CODE": "ACCOUNTNUM",
    "MEDICAL_LICENSE": "IDCARDNUM",
    "IN_VEHICLE_REGISTRATION": "IDCARDNUM"
}

def load_ai4privacy_dataset(num_samples: int = 100, split: str = "validation") -> List[Dict]:
    """
    Load and prepare samples from the ai4privacy dataset.
    
    Args:
        num_samples: Number of samples to evaluate
        split: Dataset split to use ("train" or "validation")
    
    Returns:
        List of prepared test cases
    """
    # Load the dataset
    dataset = load_dataset("ai4privacy/pii-masking-400k")
    
    # Get the specified split
    data_split = dataset[split]
    
    # Randomly sample entries if num_samples is less than total
    if num_samples < len(data_split):
        indices = random.sample(range(len(data_split)), num_samples)
        samples = [data_split[i] for i in indices]
    else:
        samples = data_split
    
    # Convert to test case format
    test_cases = []
    for sample in samples:
        # Extract entities from privacy_mask
        entities: Dict[str, List[str]] = {}
        for entity in sample['privacy_mask']:
            label = entity['label']
            value = entity['value']
            if label not in entities:
                entities[label] = []
            entities[label].append(value)
        
        test_case = {
            "description": f"AI4Privacy Sample (ID: {sample['uid']})",
            "input_text": sample['source_text'],
            "expected_entities": entities,
            "masked_text": sample['masked_text'],
            "language": sample['language'],
            "locale": sample['locale']
        }
        test_cases.append(test_case)
    
    return test_cases

@weave.op()
def evaluate_model(guardrail, test_cases: List[Dict]) -> Tuple[Dict, List[Dict]]:
    """
    Evaluate a model on the test cases.
    
    Args:
        guardrail: Entity recognition guardrail to evaluate
        test_cases: List of test cases
    
    Returns:
        Tuple of (metrics dict, detailed results list)
    """
    metrics = {
        "total": len(test_cases),
        "passed": 0,
        "failed": 0,
        "entity_metrics": {}  # Will store precision/recall per entity type
    }
    
    detailed_results = []
    
    for test_case in tqdm(test_cases, desc="Evaluating samples"):
        # Run detection
        result = guardrail.guard(test_case['input_text'])
        detected = result.detected_entities
        expected = test_case['expected_entities']
        
        # Map Presidio entities if this is the Presidio guardrail
        if isinstance(guardrail, PresidioEntityRecognitionGuardrail):
            mapped_detected = {}
            for entity_type, values in detected.items():
                mapped_type = PRESIDIO_TO_TRANSFORMER_MAPPING.get(entity_type)
                if mapped_type:
                    if mapped_type not in mapped_detected:
                        mapped_detected[mapped_type] = []
                    mapped_detected[mapped_type].extend(values)
            detected = mapped_detected
        
        # Track entity-level metrics
        all_entity_types = set(list(detected.keys()) + list(expected.keys()))
        entity_results = {}
        
        for entity_type in all_entity_types:
            detected_set = set(detected.get(entity_type, []))
            expected_set = set(expected.get(entity_type, []))
            
            # Calculate metrics
            true_positives = len(detected_set & expected_set)
            false_positives = len(detected_set - expected_set)
            false_negatives = len(expected_set - detected_set)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            entity_results[entity_type] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives
            }
            
            # Aggregate metrics
            if entity_type not in metrics["entity_metrics"]:
                metrics["entity_metrics"][entity_type] = {
                    "total_true_positives": 0,
                    "total_false_positives": 0,
                    "total_false_negatives": 0
                }
            metrics["entity_metrics"][entity_type]["total_true_positives"] += true_positives
            metrics["entity_metrics"][entity_type]["total_false_positives"] += false_positives
            metrics["entity_metrics"][entity_type]["total_false_negatives"] += false_negatives
        
        # Store detailed result
        detailed_result = {
            "id": test_case.get("description", ""),
            "language": test_case.get("language", ""),
            "locale": test_case.get("locale", ""),
            "input_text": test_case["input_text"],
            "expected_entities": expected,
            "detected_entities": detected,
            "entity_metrics": entity_results,
            "anonymized_text": result.anonymized_text if result.anonymized_text else None
        }
        detailed_results.append(detailed_result)
        
        # Update pass/fail counts
        if all(entity_results[et]["f1"] == 1.0 for et in entity_results):
            metrics["passed"] += 1
        else:
            metrics["failed"] += 1
    
    # Calculate final entity metrics and track totals for overall metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for entity_type, counts in metrics["entity_metrics"].items():
        tp = counts["total_true_positives"]
        fp = counts["total_false_positives"]
        fn = counts["total_false_negatives"]
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics["entity_metrics"][entity_type].update({
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
    
    # Calculate overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    metrics["overall"] = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn
    }
    
    return metrics, detailed_results

def save_results(metrics: Dict, detailed_results: List[Dict], model_name: str, output_dir: str = "evaluation_results"):
    """Save evaluation results to files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save metrics summary
    with open(output_dir / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed results
    with open(output_dir / f"{model_name}_detailed_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)

def print_metrics_summary(metrics: Dict):
    """Print a summary of the evaluation metrics"""
    print("\nEvaluation Summary")
    print("=" * 80)
    print(f"Total Samples: {metrics['total']}")
    print(f"Passed: {metrics['passed']}")
    print(f"Failed: {metrics['failed']}")
    print(f"Success Rate: {(metrics['passed']/metrics['total'])*100:.1f}%")
    
    # Print overall metrics
    print("\nOverall Metrics:")
    print("-" * 80)
    print(f"{'Metric':<20} {'Value':>10}")
    print("-" * 80)
    print(f"{'Precision':<20} {metrics['overall']['precision']:>10.2f}")
    print(f"{'Recall':<20} {metrics['overall']['recall']:>10.2f}")
    print(f"{'F1':<20} {metrics['overall']['f1']:>10.2f}")
    
    print("\nEntity-level Metrics:")
    print("-" * 80)
    print(f"{'Entity Type':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 80)
    for entity_type, entity_metrics in metrics["entity_metrics"].items():
        print(f"{entity_type:<20} {entity_metrics['precision']:>10.2f} {entity_metrics['recall']:>10.2f} {entity_metrics['f1']:>10.2f}")

def main():
    """Main evaluation function"""
    weave.init("guardrails-genie-pii-evaluation-demo")
    
    # Load test cases
    test_cases = load_ai4privacy_dataset(num_samples=100)
    
    # Initialize models to evaluate
    models = {
        "regex": RegexEntityRecognitionGuardrail(should_anonymize=True, show_available_entities=True),
        "presidio": PresidioEntityRecognitionGuardrail(should_anonymize=True, show_available_entities=True),
        "transformers": TransformersEntityRecognitionGuardrail(should_anonymize=True, show_available_entities=True)
    }
    
    # Evaluate each model
    for model_name, guardrail in models.items():
        print(f"\nEvaluating {model_name} model...")
        metrics, detailed_results = evaluate_model(guardrail, test_cases)
        
        # Print and save results
        print_metrics_summary(metrics)
        save_results(metrics, detailed_results, model_name)

if __name__ == "__main__":
    from guardrails_genie.guardrails.entity_recognition.regex_entity_recognition_guardrail import RegexEntityRecognitionGuardrail
    from guardrails_genie.guardrails.entity_recognition.presidio_entity_recognition_guardrail import PresidioEntityRecognitionGuardrail
    from guardrails_genie.guardrails.entity_recognition.transformers_entity_recognition_guardrail import TransformersEntityRecognitionGuardrail
    
    main()