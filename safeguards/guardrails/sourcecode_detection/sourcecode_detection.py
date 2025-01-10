from typing import Any

import weave
from pydantic import PrivateAttr

try:
    import torch
    from transformers import pipeline
except ImportError:
    import_failed = True
    print(
        "The `transformers` package is required to use the SourcecodeDetectionGuardrail, please run `pip install transformers`"
    )
from safeguards.guardrails.base import Guardrail


class SourceCodeDetectionGuardrail(Guardrail):
    """
    A guardrail that uses a pre-trained text-classification model to classify prompts
     to detect source code within the prompts.

    Attributes:
        device (str): The device to run the model on, default is 'cpu'.
        model_name_or_path (str): The path or name of the pre-trained model.
        _classifier (Any): The classifier pipeline for text classification.
        _label2id (dict[str, int]): A dictionary mapping labels to IDs.
    """

    device: str = "cpu"
    model_name_or_path: str = "wandb/sourcecode-detection"
    _classifier: Any = PrivateAttr()
    _label2id: dict[str, int] = PrivateAttr()

    def model_post_init(self, __context) -> None:
        if not torch.cuda.is_available() and "cuda" in self.device:
            raise ValueError("CUDA is not available")
        self._classifier = pipeline(
            task="text-classification",
            model=self.model_name_or_path,
            device=self.device,
        )
        self._label2id = {"no_code": 0, "code": 1}

    @weave.op
    def score_texts(self, text: str) -> dict[str, Any]:
        """
        Scores the given text to determine if it contains source code.

        Args:
            text (str): The text to be scored.

        Returns:
            dict[str, Any]: A dictionary containing the confidence score and a boolean indicating if the text has code.
        """
        output = self._classifier(inputs={"text": text})
        return {
            "confidence": round(output["score"] * 100, 2),
            "has_code": bool(self._label2id.get(output["label"], -1)),
        }

    @weave.op()
    def guard(self, prompt: str) -> dict[str, Any]:
        """
        Guards the given prompt by scoring it and determining if it is safe.

        Args:
            prompt (str): The prompt to be guarded.

        Returns:
            dict: A dictionary containing the safety status and a summary of the result.
        """
        response = self.score_texts(prompt)
        return {
            "safe": response["has_code"] == 0,
            "summary": "Prompt is deemed to {result} with {confidence}% confidence.".format(
                **{
                    "result": (
                        "have code" if response["has_code"] == 1 else "not have code"
                    ),
                    "confidence": response["confidence"],
                }
            ),
        }

    @weave.op()
    def predict(self, prompt: str) -> dict[str, Any]:
        """
        Predicts the safety of the given prompt.

        Args:
            prompt (str): The prompt to be predicted.

        Returns:
            dict: The result of the guard method.
        """
        return self.guard(prompt)
