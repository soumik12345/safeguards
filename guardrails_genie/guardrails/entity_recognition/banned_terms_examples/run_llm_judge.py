from guardrails_genie.guardrails.entity_recognition.llm_judge_entity_recognition_guardrail import RestrictedTermsJudge
from guardrails_genie.guardrails.entity_recognition.banned_terms_examples.banned_term_examples import (
    RESTRICTED_TERMS_EXAMPLES, 
    EDGE_CASE_EXAMPLES, 
    run_test_case
)
from guardrails_genie.llm import OpenAIModel
import weave

def test_restricted_terms_detection():
    """Test restricted terms detection scenarios using predefined test cases"""
    weave.init("guardrails-genie-restricted-terms-llm-judge")
    
    # Create the guardrail with OpenAI model
    llm_judge = RestrictedTermsJudge(
        should_anonymize=True,
        llm_model=OpenAIModel()
    )

    # Test statistics
    total_tests = len(RESTRICTED_TERMS_EXAMPLES) + len(EDGE_CASE_EXAMPLES)
    passed_tests = 0

    # Test main restricted terms examples
    print("\nRunning Main Restricted Terms Tests")
    print("=" * 80)
    for test_case in RESTRICTED_TERMS_EXAMPLES:
        if run_test_case(llm_judge, test_case):
            passed_tests += 1

    # Test edge cases
    print("\nRunning Edge Cases")
    print("=" * 80)
    for test_case in EDGE_CASE_EXAMPLES:
        if run_test_case(llm_judge, test_case, "Edge"):
            passed_tests += 1

    # Print summary
    print("\nTest Summary")
    print("=" * 80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")

if __name__ == "__main__":
    test_restricted_terms_detection()
