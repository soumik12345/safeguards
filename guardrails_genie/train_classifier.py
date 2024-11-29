import evaluate
import numpy as np
import streamlit as st
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerControl, TrainerState

import wandb


class StreamlitProgressbarCallback(TrainerCallback):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_bar = st.progress(0, text="Training")

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        super().on_step_begin(args, state, control, **kwargs)
        self.progress_bar.progress(
            (state.global_step * 100 // state.max_steps) + 1,
            text=f"Training {state.global_step} / {state.max_steps}",
        )


def train_binary_classifier(
    project_name: str,
    entity_name: str,
    run_name: str,
    dataset_repo: str = "geekyrakshit/prompt-injection-dataset",
    model_name: str = "distilbert/distilbert-base-uncased",
    prompt_column_name: str = "prompt",
    learning_rate: float = 1e-5,
    batch_size: int = 16,
    num_epochs: int = 2,
    weight_decay: float = 0.01,
    eval_steps: int = 2000,
    save_steps: int = 2000,
    streamlit_mode: bool = False,
):
    wandb.init(project=project_name, entity=entity_name, name=run_name)
    if streamlit_mode:
        st.markdown(
            f"Explore your training logs on [Weights & Biases]({wandb.run.url})"
        )
    dataset = load_dataset(dataset_repo)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_datasets = dataset.map(
        lambda examples: tokenizer(examples[prompt_column_name], truncation=True),
        batched=True,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    id2label = {0: "SAFE", 1: "INJECTION"}
    label2id = {"SAFE": 0, "INJECTION": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="binary-classifier",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to="wandb",
            logging_strategy="steps",
            logging_steps=1,
        ),
        train_dataset=tokenized_datasets["train"].shuffle(seed=42),
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[StreamlitProgressbarCallback()] if streamlit_mode else [],
    )
    try:
        training_output = trainer.train()
    except Exception as e:
        wandb.finish()
        raise e
    wandb.finish()
    return training_output
