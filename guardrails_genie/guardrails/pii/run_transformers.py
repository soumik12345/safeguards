from guardrails_genie.guardrails.pii.transformers_pipeline_guardrail import TransformersPipelinePIIGuardrail
import weave

def run_transformers_pipeline():
    weave.init("guardrails-genie-pii-transformers-pipeline-model")
    
    # Create the guardrail with default entities and anonymization enabled
    pii_guardrail = TransformersPipelinePIIGuardrail(
        selected_entities=["GIVENNAME", "SURNAME", "EMAIL", "TELEPHONENUM", "SOCIALNUM", "PHONE_NUMBER"],
        should_anonymize=True,
        model_name="lakshyakh93/deberta_finetuned_pii",
        show_available_entities=True
    )

    # Check a prompt
    prompt = "Please contact John Smith at john.smith@email.com or call 123-456-7890. My SSN is 123-45-6789"
    result = pii_guardrail.guard(prompt, aggregate_redaction=False)
    print(result)

    # Result will contain:
    # - contains_pii: True
    # - detected_pii_types: {
    #     "GIVENNAME": ["John"],
    #     "SURNAME": ["Smith"],
    #     "EMAIL": ["john.smith@email.com"],
    #     "TELEPHONENUM": ["123-456-7890"],
    #     "SOCIALNUM": ["123-45-6789"]
    # }
    # - safe_to_process: False
    # - explanation: Detailed explanation of findings
    # - anonymized_text: "Please contact [redacted] [redacted] at [redacted] or call [redacted]. My SSN is [redacted]"


if __name__ == "__main__":
    run_transformers_pipeline()
