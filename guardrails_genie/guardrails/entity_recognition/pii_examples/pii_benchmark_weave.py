import asyncio
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import weave
from datasets import load_dataset
from weave import Evaluation
from weave.scorers import Scorer

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
    "IN_VEHICLE_REGISTRATION": "IDCARDNUM",
}


class EntityRecognitionScorer(Scorer):
    """Scorer for evaluating entity recognition performance"""

    @weave.op()
    async def score(
        self, model_output: Optional[dict], input_text: str, expected_entities: Dict
    ) -> Dict:
        """Score entity recognition results"""
        if not model_output:
            return {"f1": 0.0}

        # Convert Pydantic model to dict if necessary
        if hasattr(model_output, "model_dump"):
            model_output = model_output.model_dump()
        elif hasattr(model_output, "dict"):
            model_output = model_output.dict()

        detected = model_output.get("detected_entities", {})

        # Map Presidio entities if needed
        if model_output.get("model_type") == "presidio":
            mapped_detected = {}
            for entity_type, values in detected.items():
                mapped_type = PRESIDIO_TO_TRANSFORMER_MAPPING.get(entity_type)
                if mapped_type:
                    if mapped_type not in mapped_detected:
                        mapped_detected[mapped_type] = []
                    mapped_detected[mapped_type].extend(values)
            detected = mapped_detected

        # Track entity-level metrics
        all_entity_types = set(list(detected.keys()) + list(expected_entities.keys()))
        entity_metrics = {}

        for entity_type in all_entity_types:
            detected_set = set(detected.get(entity_type, []))
            expected_set = set(expected_entities.get(entity_type, []))

            # Calculate metrics
            true_positives = len(detected_set & expected_set)
            false_positives = len(detected_set - expected_set)
            false_negatives = len(expected_set - detected_set)

            if entity_type not in entity_metrics:
                entity_metrics[entity_type] = {
                    "total_true_positives": 0,
                    "total_false_positives": 0,
                    "total_false_negatives": 0,
                }

            entity_metrics[entity_type]["total_true_positives"] += true_positives
            entity_metrics[entity_type]["total_false_positives"] += false_positives
            entity_metrics[entity_type]["total_false_negatives"] += false_negatives

            # Calculate per-entity metrics
            precision = (
                true_positives / (true_positives + false_positives)
                if (true_positives + false_positives) > 0
                else 0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            entity_metrics[entity_type].update(
                {"precision": precision, "recall": recall, "f1": f1}
            )

        # Calculate overall metrics
        total_tp = sum(
            metrics["total_true_positives"] for metrics in entity_metrics.values()
        )
        total_fp = sum(
            metrics["total_false_positives"] for metrics in entity_metrics.values()
        )
        total_fn = sum(
            metrics["total_false_negatives"] for metrics in entity_metrics.values()
        )

        overall_precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        )
        overall_recall = (
            total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        )
        overall_f1 = (
            2
            * (overall_precision * overall_recall)
            / (overall_precision + overall_recall)
            if (overall_precision + overall_recall) > 0
            else 0
        )

        entity_metrics["overall"] = {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
            "total_true_positives": total_tp,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
        }

        return entity_metrics


def load_ai4privacy_dataset(
    num_samples: int = 100, split: str = "validation"
) -> List[Dict]:
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
        for entity in sample["privacy_mask"]:
            label = entity["label"]
            value = entity["value"]
            if label not in entities:
                entities[label] = []
            entities[label].append(value)

        test_case = {
            "description": f"AI4Privacy Sample (ID: {sample['uid']})",
            "input_text": sample["source_text"],
            "expected_entities": entities,
            "masked_text": sample["masked_text"],
            "language": sample["language"],
            "locale": sample["locale"],
        }
        test_cases.append(test_case)

    return test_cases


def save_results(
    weave_results: Dict, model_name: str, output_dir: str = "evaluation_results"
):
    """Save evaluation results to files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Extract and process results
    scorer_results = weave_results.get("EntityRecognitionScorer", [])
    if not scorer_results or all(r is None for r in scorer_results):
        print(f"No valid results to save for {model_name}")
        return

    # Calculate summary metrics
    total_samples = len(scorer_results)
    passed = sum(1 for r in scorer_results if r is not None and not isinstance(r, str))

    # Aggregate entity-level metrics
    entity_metrics = {}
    for result in scorer_results:
        try:
            if isinstance(result, str) or not result:
                continue

            for entity_type, metrics in result.items():
                if entity_type not in entity_metrics:
                    entity_metrics[entity_type] = {
                        "precision": [],
                        "recall": [],
                        "f1": [],
                    }
                entity_metrics[entity_type]["precision"].append(metrics["precision"])
                entity_metrics[entity_type]["recall"].append(metrics["recall"])
                entity_metrics[entity_type]["f1"].append(metrics["f1"])
        except (AttributeError, TypeError, KeyError):
            continue

    # Calculate averages
    summary_metrics = {
        "total": total_samples,
        "passed": passed,
        "failed": total_samples - passed,
        "success_rate": (passed / total_samples) if total_samples > 0 else 0,
        "entity_metrics": {
            entity_type: {
                "precision": (
                    sum(metrics["precision"]) / len(metrics["precision"])
                    if metrics["precision"]
                    else 0
                ),
                "recall": (
                    sum(metrics["recall"]) / len(metrics["recall"])
                    if metrics["recall"]
                    else 0
                ),
                "f1": sum(metrics["f1"]) / len(metrics["f1"]) if metrics["f1"] else 0,
            }
            for entity_type, metrics in entity_metrics.items()
        },
    }

    # Save files
    with open(output_dir / f"{model_name}_metrics.json", "w") as f:
        json.dump(summary_metrics, f, indent=2)

    # Save detailed results, filtering out string results
    detailed_results = [
        r for r in scorer_results if not isinstance(r, str) and r is not None
    ]
    with open(output_dir / f"{model_name}_detailed_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)


def print_metrics_summary(weave_results: Dict):
    """Print a summary of the evaluation metrics"""
    print("\nEvaluation Summary")
    print("=" * 80)

    # Extract results from Weave's evaluation format
    scorer_results = weave_results.get("EntityRecognitionScorer", {})
    if not scorer_results:
        print("No valid results available")
        return

    # Calculate overall metrics
    total_samples = int(weave_results.get("model_latency", {}).get("count", 0))
    passed = total_samples  # Since we have results, all samples passed
    failed = 0

    print(f"Total Samples: {total_samples}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/total_samples)*100:.2f}%")

    # Print overall metrics
    if "overall" in scorer_results:
        overall = scorer_results["overall"]
        print("\nOverall Metrics:")
        print("-" * 80)
        print(f"{'Metric':<20} {'Value':>10}")
        print("-" * 80)
        print(f"{'Precision':<20} {overall['precision']['mean']:>10.2f}")
        print(f"{'Recall':<20} {overall['recall']['mean']:>10.2f}")
        print(f"{'F1':<20} {overall['f1']['mean']:>10.2f}")

    # Print entity-level metrics
    print("\nEntity-Level Metrics:")
    print("-" * 80)
    print(f"{'Entity Type':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 80)

    for entity_type, metrics in scorer_results.items():
        if entity_type == "overall":
            continue

        precision = metrics.get("precision", {}).get("mean", 0)
        recall = metrics.get("recall", {}).get("mean", 0)
        f1 = metrics.get("f1", {}).get("mean", 0)

        print(f"{entity_type:<20} {precision:>10.2f} {recall:>10.2f} {f1:>10.2f}")


def preprocess_model_input(example: Dict) -> Dict:
    """Preprocess dataset example to match model input format."""
    return {
        "prompt": example["input_text"],
        "model_type": example.get(
            "model_type", "unknown"
        ),  # Add model type for Presidio mapping
    }


def main():
    """Main evaluation function"""
    weave.init("guardrails-genie-pii-evaluation")

    # Load test cases
    test_cases = load_ai4privacy_dataset(num_samples=100)

    # Add model type to test cases for Presidio mapping
    models = {
        # "regex": RegexEntityRecognitionGuardrail(should_anonymize=True),
        "presidio": PresidioEntityRecognitionGuardrail(should_anonymize=True),
        # "transformers": TransformersEntityRecognitionGuardrail(should_anonymize=True)
    }

    scorer = EntityRecognitionScorer()

    # Evaluate each model
    for model_name, guardrail in models.items():
        print(f"\nEvaluating {model_name} model...")
        # Add model type to test cases
        model_test_cases = [{**case, "model_type": model_name} for case in test_cases]

        evaluation = Evaluation(
            dataset=model_test_cases,
            scorers=[scorer],
            preprocess_model_input=preprocess_model_input,
        )

        asyncio.run(evaluation.evaluate(guardrail))


if __name__ == "__main__":
    from guardrails_genie.guardrails.entity_recognition.presidio_entity_recognition_guardrail import (
        PresidioEntityRecognitionGuardrail,
    )

    main()
