import json
from typing import Optional

import weave
from rich.progress import track

from .trace_utils import serialize_input_output_objects


class EvaluationClassifier:

    def __init__(self, project: str, call_id: str) -> None:
        self.base_call = weave.init(project).get_call(call_id=call_id)
        self.predict_and_score_calls = []

    def _get_call_name_from_op_name(self, op_name: str) -> str:
        return op_name.split("/")[-1].split(":")[0]

    def register_predict_and_score_calls(
        self,
        failure_condition: str,
        max_predict_and_score_calls: Optional[int] = None,
        save_filepath: Optional[str] = None,
    ):
        count_traces_parsed = 0
        for predict_and_score_call in track(
            self.base_call.children(),
            description="Parsing predict and score calls",
            total=max_predict_and_score_calls - 1,
        ):
            if "Evaluation.summarize" in predict_and_score_call._op_name:
                break
            elif "Evaluation.predict_and_score" in predict_and_score_call._op_name:
                if eval(
                    "serialize_input_output_objects(predict_and_score_call.output)['scores']"
                    + failure_condition
                ):
                    self.predict_and_score_calls.append(
                        self.parse_call(predict_and_score_call)
                    )
                count_traces_parsed += 1
                if count_traces_parsed == max_predict_and_score_calls:
                    break
        if len(self.predict_and_score_calls) > 0 and save_filepath is not None:
            self.save_calls(save_filepath)

    def parse_call(self, child_call) -> dict:
        call_dict = {
            "call_name": self._get_call_name_from_op_name(child_call._op_name),
            "inputs": serialize_input_output_objects(child_call.inputs),
            "outputs": serialize_input_output_objects(child_call.output),
            "child_calls": [self.parse_call(child) for child in child_call.children()],
        }
        return call_dict

    def save_calls(self, filepath: str):
        with open(filepath, "w") as file:
            json.dump(self.predict_and_score_calls, file, indent=4)
