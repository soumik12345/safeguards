import weave

from safeguards.guardrails.entity_recognition.pii_examples.pii_test_examples import (
    EDGE_CASE_EXAMPLES,
    PII_TEST_EXAMPLES,
    run_test_case,
)
from safeguards.guardrails.entity_recognition.regex_entity_recognition_guardrail import (
    RegexEntityRecognitionGuardrail,
)


def test_pii_detection():
    """Test PII detection scenarios using predefined test cases"""
    weave.init("guardrails-genie-pii-regex-model")

    # Create the guardrail with default entities and anonymization enabled
    pii_guardrail = RegexEntityRecognitionGuardrail(
        should_anonymize=True, show_available_entities=True
    )

    # Test statistics
    total_tests = len(PII_TEST_EXAMPLES) + len(EDGE_CASE_EXAMPLES)
    passed_tests = 0

    # Test main PII examples
    print("\nRunning Main PII Tests")
    print("=" * 80)
    for test_case in PII_TEST_EXAMPLES:
        if run_test_case(pii_guardrail, test_case):
            passed_tests += 1

    # Test edge cases
    print("\nRunning Edge Cases")
    print("=" * 80)
    for test_case in EDGE_CASE_EXAMPLES:
        if run_test_case(pii_guardrail, test_case, "Edge"):
            passed_tests += 1

    # Print summary
    print("\nTest Summary")
    print("=" * 80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")


if __name__ == "__main__":
    test_pii_detection()
