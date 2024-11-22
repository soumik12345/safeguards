from guardrails_genie.guardrails.pii.regex_pii_guardrail import RegexPIIGuardrail
import weave

def run_regex_model():
    weave.init("guardrails-genie-pii-regex-model")
    # Create the guardrail
    pii_guardrail = RegexPIIGuardrail(use_defaults=True)

    # Check a prompt
    prompt = "Please contact john.doe@email.com or call 123-456-7890"
    result = pii_guardrail.guard(prompt)
    print(result)

    # Result will contain:
    # - contains_pii: True
    # - detected_pii_types: {"email": ["john.doe@email.com"], "phone_number": ["123-456-7890"]}
    # - safe_to_process: False
    # - explanation: Detailed explanation of findings

if __name__ == "__main__":
    run_regex_model()
