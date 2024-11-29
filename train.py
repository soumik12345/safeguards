from dotenv import load_dotenv

from guardrails_genie.train_classifier import train_binary_classifier

load_dotenv()
train_binary_classifier(
    project_name="guardrails-genie",
    entity_name="geekyrakshit",
    model_name="distilbert/distilbert-base-uncased",
    run_name="distilbert/distilbert-base-uncased-finetuned",
    dataset_repo="jayavibhav/prompt-injection",
    prompt_column_name="text",
)
