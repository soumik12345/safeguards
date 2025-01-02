import weave
from rich.progress import track

from .trace_utils import serialize_input_output_objects


class EvaluationTraceParser:

    def __init__(self, project: str, call_id: str) -> None:
        self.base_call = weave.init(project).get_call(call_id=call_id)
        self.predict_and_score_calls = []

    def _get_call_name_from_op_name(self, op_name: str) -> str:
        return op_name.split("/")[-1].split(":")[0]

    def register_predict_and_score_calls(self):
        for predict_and_score_call in track(
            self.base_call.children(), description="Parsing predict and score calls"
        ):
            if "Evaluation.summarize" in predict_and_score_call._op_name:
                break
            elif "Evaluation.predict_and_score" in predict_and_score_call._op_name:
                self.predict_and_score_calls.append(
                    self.parse_call(predict_and_score_call)
                )
            break

    def parse_call(self, child_call) -> dict:
        call_dict = {
            "call_name": self._get_call_name_from_op_name(child_call._op_name),
            "inputs": serialize_input_output_objects(child_call.inputs),
            "outputs": serialize_input_output_objects(child_call.output),
            "child_calls": [self.parse_call(child) for child in child_call.children()],
        }
        return call_dict
