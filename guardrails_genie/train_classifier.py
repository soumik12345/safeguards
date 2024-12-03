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
    """
    StreamlitProgressbarCallback is a custom callback for the Hugging Face Trainer
    that integrates a progress bar into a Streamlit application. This class updates
    the progress bar at each training step, providing real-time feedback on the
    training process within the Streamlit interface.

    Attributes:
        progress_bar (streamlit.delta_generator.DeltaGenerator): A Streamlit progress
            bar object initialized to 0 with the text "Training".

    Methods:
        on_step_begin(args, state, control, **kwargs):
            Updates the progress bar at the beginning of each training step. The progress
            is calculated as the percentage of completed steps out of the total steps.
            The progress bar text is updated to show the current step and the total steps.
    """

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
    id2label: dict[int, str] = {0: "SAFE", 1: "INJECTION"},
    label2id: dict[str, int] = {"SAFE": 0, "INJECTION": 1},
    learning_rate: float = 1e-5,
    batch_size: int = 16,
    num_epochs: int = 2,
    weight_decay: float = 0.01,
    save_steps: int = 1000,
    streamlit_mode: bool = False,
):
    """
    Trains a binary classifier using a specified dataset and model architecture.

    This function sets up and trains a binary sequence classification model using
    the Hugging Face Transformers library. It integrates with Weights & Biases for
    experiment tracking and optionally displays a progress bar in a Streamlit app.

    Args:
        project_name (str): The name of the Weights & Biases project.
        entity_name (str): The Weights & Biases entity (user or team).
        run_name (str): The name of the Weights & Biases run.
        dataset_repo (str, optional): The Hugging Face dataset repository to load.
            Defaults to "geekyrakshit/prompt-injection-dataset".
        model_name (str, optional): The pre-trained model to use. Defaults to
            "distilbert/distilbert-base-uncased".
        prompt_column_name (str, optional): The column name in the dataset containing
            the text prompts. Defaults to "prompt".
        id2label (dict[int, str], optional): Mapping from label IDs to label names.
            Defaults to {0: "SAFE", 1: "INJECTION"}.
        label2id (dict[str, int], optional): Mapping from label names to label IDs.
            Defaults to {"SAFE": 0, "INJECTION": 1}.
        learning_rate (float, optional): The learning rate for training. Defaults to 1e-5.
        batch_size (int, optional): The batch size for training and evaluation.
            Defaults to 16.
        num_epochs (int, optional): The number of training epochs. Defaults to 2.
        weight_decay (float, optional): The weight decay for the optimizer. Defaults to 0.01.
        save_steps (int, optional): The number of steps between model checkpoints.
            Defaults to 1000.
        streamlit_mode (bool, optional): If True, integrates with Streamlit to display
            a progress bar. Defaults to False.

    Returns:
        dict: The output of the training process, including metrics and model state.

    Raises:
        Exception: If an error occurs during training, the exception is raised after
            ensuring Weights & Biases run is finished.
    """
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
            eval_strategy="epoch",
            save_strategy="steps",
            save_steps=save_steps,
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to="wandb",
            logging_strategy="steps",
            logging_steps=1,
        ),
        train_dataset=tokenized_datasets["train"],
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
