import os
import shutil
from glob import glob
from typing import Optional

import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from pydantic import BaseModel
from rich.progress import track
from safetensors.torch import load_model, save_model
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import wandb


class DatasetArgs(BaseModel):
    dataset_address: str
    train_dataset_range: int
    test_dataset_range: int


class LlamaGuardFineTuner:
    """
    `LlamaGuardFineTuner` is a class designed to fine-tune and evaluate the
    [Prompt Guard model by Meta LLama](meta-llama/Prompt-Guard-86M) for prompt
    classification tasks, specifically for detecting prompt injection attacks. It
    integrates with Weights & Biases for experiment tracking and optionally
    displays progress in a Streamlit app.

    !!! example "Sample Usage"
        ```python
        from guardrails_genie.train.llama_guard import LlamaGuardFineTuner, DatasetArgs

        fine_tuner = LlamaGuardFineTuner(
            wandb_project="guardrails-genie",
            wandb_entity="geekyrakshit",
            streamlit_mode=False,
        )
        fine_tuner.load_dataset(
            DatasetArgs(
                dataset_address="wandb/synthetic-prompt-injections",
                train_dataset_range=-1,
                test_dataset_range=-1,
            )
        )
        fine_tuner.load_model()
        fine_tuner.train(save_interval=100)
        ```

    Args:
        wandb_project (str): The name of the Weights & Biases project.
        wandb_entity (str): The Weights & Biases entity (user or team).
        streamlit_mode (bool): If True, integrates with Streamlit to display progress.
    """

    def __init__(
        self, wandb_project: str, wandb_entity: str, streamlit_mode: bool = False
    ):
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.streamlit_mode = streamlit_mode

    def load_dataset(self, dataset_args: DatasetArgs):
        """
        Loads the training and testing datasets based on the provided dataset arguments.

        This function uses the `load_dataset` function from the `datasets` library to load
        the dataset specified by the `dataset_address` attribute of the `dataset_args` parameter.
        It then selects a subset of the training and testing datasets based on the specified
        ranges in `train_dataset_range` and `test_dataset_range` attributes of `dataset_args`.
        If the specified range is less than or equal to 0 or exceeds the length of the dataset,
        the entire dataset is used.

        Args:
            dataset_args (DatasetArgs): An instance of the `DatasetArgs` class containing
                the dataset address and the ranges for training and testing datasets.

        Attributes:
            train_dataset: The selected training dataset.
            test_dataset: The selected testing dataset.
        """
        self.dataset_args = dataset_args
        dataset = load_dataset(dataset_args.dataset_address)
        self.train_dataset = (
            dataset["train"]
            if dataset_args.train_dataset_range <= 0
            or dataset_args.train_dataset_range > len(dataset["train"])
            else dataset["train"].select(range(dataset_args.train_dataset_range))
        )
        self.test_dataset = (
            dataset["test"]
            if dataset_args.test_dataset_range <= 0
            or dataset_args.test_dataset_range > len(dataset["test"])
            else dataset["test"].select(range(dataset_args.test_dataset_range))
        )

    def load_model(
        self,
        model_name: str = "meta-llama/Prompt-Guard-86M",
        checkpoint: Optional[str] = None,
    ):
        """
        Loads the specified pre-trained model and tokenizer for sequence classification tasks.

        This function sets the device to GPU if available, otherwise defaults to CPU. It then
        loads the tokenizer and model from the Hugging Face model hub using the provided model name.
        The model is moved to the specified device (GPU or CPU).

        Args:
            model_name (str): The name of the pre-trained model to load.

        Attributes:
            device (str): The device to run the model on, either "cuda" for GPU or "cpu".
            model_name (str): The name of the loaded pre-trained model.
            tokenizer (AutoTokenizer): The tokenizer associated with the pre-trained model.
            model (AutoModelForSequenceClassification): The loaded pre-trained model for sequence classification.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if checkpoint is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name
            ).to(self.device)
        else:
            api = wandb.Api()
            artifact = api.artifact(checkpoint.removeprefix("wandb://"))
            artifact_dir = artifact.download()
            model_file_path = glob(os.path.join(artifact_dir, "model-*.safetensors"))[0]
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, 2)
            self.model.num_labels = 2
            load_model(self.model, model_file_path)
            self.model = self.model.to(self.device)

    def show_dataset_sample(self):
        """
        Displays a sample of the training and testing datasets using Streamlit.

        This function checks if the `streamlit_mode` attribute is enabled. If it is,
        it converts the training and testing datasets to pandas DataFrames and displays
        the first few rows of each dataset using Streamlit's `dataframe` function. The
        training dataset sample is displayed under the heading "Train Dataset Sample",
        and the testing dataset sample is displayed under the heading "Test Dataset Sample".

        Note:
            This function requires the `streamlit` library to be installed and the
            `streamlit_mode` attribute to be set to True.
        """
        if self.streamlit_mode:
            st.markdown("### Train Dataset Sample")
            st.dataframe(self.train_dataset.to_pandas().head())
            st.markdown("### Test Dataset Sample")
            st.dataframe(self.test_dataset.to_pandas().head())

    def evaluate_batch(
        self,
        texts,
        batch_size: int = 32,
        positive_label: int = 2,
        temperature: float = 1.0,
        truncation: bool = True,
        max_length: int = 512,
    ) -> list[float]:
        self.model.eval()
        encoded_texts = self.tokenizer(
            texts,
            padding=True,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt",
        )
        dataset = torch.utils.data.TensorDataset(
            encoded_texts["input_ids"], encoded_texts["attention_mask"]
        )
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        scores = []
        progress_bar = (
            st.progress(0, text="Evaluating") if self.streamlit_mode else None
        )
        for i, batch in track(
            enumerate(data_loader), description="Evaluating", total=len(data_loader)
        ):
            input_ids, attention_mask = [b.to(self.device) for b in batch]
            with torch.no_grad():
                logits = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
            scaled_logits = logits / temperature
            probabilities = F.softmax(scaled_logits, dim=-1)
            positive_class_probabilities = (
                probabilities[:, positive_label].cpu().numpy()
            )
            scores.extend(positive_class_probabilities)
            if progress_bar:
                progress_percentage = (i + 1) * 100 // len(data_loader)
                progress_bar.progress(
                    progress_percentage,
                    text=f"Evaluating batch {i + 1}/{len(data_loader)}",
                )

        return scores

    def visualize_roc_curve(self, test_scores: list[float]):
        test_labels = [int(elt) for elt in self.test_dataset["label"]]
        fpr, tpr, _ = roc_curve(test_labels, test_scores)
        roc_auc = roc_auc_score(test_labels, test_scores)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC curve (area = {roc_auc:.3f})",
                line=dict(color="darkorange", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Guess",
                line=dict(color="navy", width=2, dash="dash"),
            )
        )
        fig.update_layout(
            title="Receiver Operating Characteristic",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0.0, 1.0]),
            yaxis=dict(range=[0.0, 1.05]),
            legend=dict(x=0.8, y=0.2),
        )
        if self.streamlit_mode:
            st.plotly_chart(fig)
        else:
            fig.show()

    def visualize_score_distribution(self, scores: list[float]):
        test_labels = [int(elt) for elt in self.test_dataset["label"]]
        positive_scores = [scores[i] for i in range(500) if test_labels[i] == 1]
        negative_scores = [scores[i] for i in range(500) if test_labels[i] == 0]
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=positive_scores,
                histnorm="probability density",
                name="Positive",
                marker_color="darkblue",
                opacity=0.75,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=negative_scores,
                histnorm="probability density",
                name="Negative",
                marker_color="darkred",
                opacity=0.75,
            )
        )
        fig.update_layout(
            title="Score Distribution for Positive and Negative Examples",
            xaxis_title="Score",
            yaxis_title="Density",
            barmode="overlay",
            legend_title="Scores",
        )
        if self.streamlit_mode:
            st.plotly_chart(fig)
        else:
            fig.show()

    def evaluate_model(
        self,
        batch_size: int = 32,
        positive_label: int = 2,
        temperature: float = 3.0,
        truncation: bool = True,
        max_length: int = 512,
    ):
        """
        Evaluates the fine-tuned model on the test dataset and visualizes the results.

        This function evaluates the model by processing the test dataset in batches.
        It computes the test scores using the `evaluate_batch` method, which takes
        several parameters to control the evaluation process, such as batch size,
        positive label, temperature, truncation, and maximum sequence length.

        After obtaining the test scores, it visualizes the performance of the model
        using two methods:
        1. `visualize_roc_curve`: Plots the Receiver Operating Characteristic (ROC) curve
           to show the trade-off between the true positive rate and false positive rate.
        2. `visualize_score_distribution`: Plots the distribution of scores for positive
           and negative examples to provide insights into the model's performance.

        Args:
            batch_size (int, optional): The number of samples to process in each batch.
            positive_label (int, optional): The label considered as positive for evaluation.
            temperature (float, optional): The temperature parameter for scaling logits.
            truncation (bool, optional): Whether to truncate sequences to the maximum length.
            max_length (int, optional): The maximum length of sequences after truncation.

        Returns:
            list[float]: The test scores obtained from the evaluation.
        """
        test_scores = self.evaluate_batch(
            self.test_dataset["text"],
            batch_size=batch_size,
            positive_label=positive_label,
            temperature=temperature,
            truncation=truncation,
            max_length=max_length,
        )
        self.visualize_roc_curve(test_scores)
        self.visualize_score_distribution(test_scores)
        return test_scores

    def collate_fn(self, batch):
        texts = [item["text"] for item in batch]
        labels = torch.tensor([int(item["label"]) for item in batch])
        encodings = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        return encodings.input_ids, encodings.attention_mask, labels

    def train(
        self,
        batch_size: int = 32,
        lr: float = 5e-6,
        num_classes: int = 2,
        log_interval: int = 1,
        save_interval: int = 50,
    ):
        """
        Fine-tunes the pre-trained LlamaGuard model on the training dataset for a single epoch.

        This function sets up and executes the training loop for the LlamaGuard model.
        It initializes the Weights & Biases (wandb) logging, configures the model's
        classifier layer to match the specified number of classes, and sets the model
        to training mode. The function uses an AdamW optimizer to update the model
        parameters based on the computed loss.

        The training process involves iterating over the training dataset in batches,
        computing the loss for each batch, and updating the model parameters. The
        function logs the loss to wandb at specified intervals and optionally displays
        a progress bar using Streamlit if `streamlit_mode` is enabled. Model checkpoints
        are saved at specified intervals during training.

        Args:
            batch_size (int, optional): The number of samples per batch during training.
            lr (float, optional): The learning rate for the optimizer.
            num_classes (int, optional): The number of output classes for the classifier.
            log_interval (int, optional): The interval (in batches) at which to log the loss.
            save_interval (int, optional): The interval (in batches) at which to save model checkpoints.

        Note:
            This function requires the `wandb` and `streamlit` libraries to be installed
            and configured appropriately.
        """
        os.makedirs("checkpoints", exist_ok=True)
        wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=f"{self.model_name}-{self.dataset_args.dataset_address.split('/')[-1]}",
            job_type="fine-tune-llama-guard",
        )
        wandb.config.dataset_args = self.dataset_args.model_dump()
        wandb.config.model_name = self.model_name
        wandb.config.batch_size = batch_size
        wandb.config.lr = lr
        wandb.config.num_classes = num_classes
        wandb.config.log_interval = log_interval
        wandb.config.save_interval = save_interval
        self.model.classifier = nn.Linear(
            self.model.classifier.in_features, num_classes
        )
        self.model.num_labels = num_classes
        self.model = self.model.to(self.device)
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        data_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        progress_bar = st.progress(0, text="Training") if self.streamlit_mode else None
        for i, batch in track(
            enumerate(data_loader), description="Training", total=len(data_loader)
        ):
            input_ids, attention_mask, labels = [x.to(self.device) for x in batch]
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % log_interval == 0:
                wandb.log({"loss": loss.item()}, step=i + 1)
            if progress_bar:
                progress_percentage = (i + 1) * 100 // len(data_loader)
                progress_bar.progress(
                    progress_percentage,
                    text=f"Training batch {i + 1}/{len(data_loader)}, Loss: {loss.item()}",
                )
            if (i + 1) % save_interval == 0 or i + 1 == len(data_loader):
                save_model(self.model, f"checkpoints/model-{i + 1}.safetensors")
                wandb.log_model(
                    f"checkpoints/model-{i + 1}.safetensors",
                    name=f"{wandb.run.id}-model",
                    aliases=f"step-{i + 1}",
                )
        wandb.finish()
        shutil.rmtree("checkpoints")
