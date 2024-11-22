from guardrails_genie.guardrails.pii.presidio_pii_guardrail import PresidioPIIGuardrail
import weave

def run_presidio_model():
    weave.init("guardrails-genie-pii-presidio-model")
    
    # Create the guardrail with default entities and anonymization enabled
    pii_guardrail = PresidioPIIGuardrail(
        selected_entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
        should_anonymize=True
    )

    # Check a prompt
    prompt = "Please contact john.doe@email.com or call 123-456-7890. My SSN is 123-45-6789"
    result = pii_guardrail.guard(prompt)
    print(result)

    # Result will contain:
    # - contains_pii: True
    # - detected_pii_types: {
    #     "EMAIL_ADDRESS": ["john.doe@email.com"],
    #     "PHONE_NUMBER": ["123-456-7890"],
    #     "US_SSN": ["123-45-6789"]
    # }
    # - safe_to_process: False
    # - explanation: Detailed explanation of findings
    # - anonymized_text: "Please contact <EMAIL_ADDRESS> or call <PHONE_NUMBER>. My SSN is <US_SSN>"

    # Example with no PII
    safe_prompt = "The weather is nice today"
    safe_result = pii_guardrail.guard(safe_prompt)
    print("\nSafe prompt result:")
    print(safe_result)

if __name__ == "__main__":
    run_presidio_model()