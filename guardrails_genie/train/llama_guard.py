import matplotlib.pyplot as plt
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
        plt.figure(figsize=(8, 6))
        test_labels = [int(elt) for elt in self.test_dataset["label"]]
        fpr, tpr, _ = roc_curve(test_labels, test_scores)
        roc_auc = roc_auc_score(test_labels, test_scores)
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.3f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        if self.streamlit_mode:
            st.pyplot(plt)
        else:
            plt.show()

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
        return test_scores
