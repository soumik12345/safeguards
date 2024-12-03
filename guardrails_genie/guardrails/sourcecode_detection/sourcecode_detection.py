from typing import Any

import weave
from pydantic import PrivateAttr

try:
    import torch
    from transformers import pipeline
except ImportError:
    import_failed = True
    print(
        "The `transformers` package is required to use the CoherenceScorer, please run `pip install transformers`"
    )
from guardrails_genie.guardrails.base import Guardrail


class SourceCodeDetector(Guardrail):
    device: str = "cpu"
    model_name_or_path: str = "wandb/sourcecode-detection"
    _classifier: Any = PrivateAttr()
    _label2id: dict[str, int] = PrivateAttr()

    def model_post_init(self, __context):
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
        output = self._classifier(inputs={"text": text})
        return {
            "confidence": round(output["score"] * 100, 2),
            "has_code": bool(self._label2id.get(output["label"], -1)),
        }

    @weave.op()
    def guard(self, prompt: str):
        response = self.score_texts(prompt)
        return {
            "safe": response["has_code"] == 0,
            "summary": "Prompt is deemed to {result} with {confidence}% confidence.".format(
                **{"result": "have code" if response["has_code"] == 1 else "not have code",
                   "confidence": response["confidence"]}),
        }

    @weave.op()
    def predict(self, prompt: str):
        return self.guard(prompt)
