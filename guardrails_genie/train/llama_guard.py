import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
from datasets import load_dataset
from pydantic import BaseModel
from rich.progress import track
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class DatasetArgs(BaseModel):
    dataset_address: str
    train_dataset_range: int
    test_dataset_range: int


class LlamaGuardFineTuner:
    def __init__(self, streamlit_mode: bool = False):
        self.streamlit_mode = streamlit_mode

    def load_dataset(self, dataset_args: DatasetArgs):
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

    def load_model(self, model_name: str = "meta-llama/Prompt-Guard-86M"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.device
        )

    def show_dataset_sample(self):
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

        # Plotting positive scores
        fig.add_trace(
            go.Histogram(
                x=positive_scores,
                histnorm="probability density",
                name="Positive",
                marker_color="darkblue",
                opacity=0.75,
            )
        )

        # Plotting negative scores
        fig.add_trace(
            go.Histogram(
                x=negative_scores,
                histnorm="probability density",
                name="Negative",
                marker_color="darkred",
                opacity=0.75,
            )
        )

        # Updating layout
        fig.update_layout(
            title="Score Distribution for Positive and Negative Examples",
            xaxis_title="Score",
            yaxis_title="Density",
            barmode="overlay",
            legend_title="Scores",
        )

        # Display the plot
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
