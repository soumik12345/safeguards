import os

import pandas as pd
import pymupdf4llm
import weave
import weave.trace
from firerequests import FireRequests


@weave.op()
def get_markdown_from_pdf_url(url: str) -> str:
    FireRequests().download(url, "temp.pdf", show_progress=False)
    markdown = pymupdf4llm.to_markdown("temp.pdf", show_progress=False)
    os.remove("temp.pdf")
    return markdown


class EvaluationCallManager:
    def __init__(self, entity: str, project: str, call_id: str, max_count: int = 10):
        self.base_call = weave.init(f"{entity}/{project}").get_call(call_id=call_id)
        self.max_count = max_count
        self.show_warning_in_app = False
        self.call_list = []

    def collect_guardrail_guard_calls_from_eval(self):
        guard_calls, count = [], 0
        for eval_predict_and_score_call in self.base_call.children():
            if "Evaluation.summarize" in eval_predict_and_score_call._op_name:
                break
            guardrail_predict_call = eval_predict_and_score_call.children()[0]
            guard_call = guardrail_predict_call.children()[0]
            score_call = eval_predict_and_score_call.children()[1]
            guard_calls.append(
                {
                    "input_prompt": str(guard_call.inputs["prompt"]),
                    "outputs": dict(guard_call.output),
                    "score": dict(score_call.output),
                }
            )
            count += 1
            if count >= self.max_count:
                self.show_warning_in_app = True
                break
        return guard_calls

    def render_calls_to_streamlit(self):
        dataframe = {
            "input_prompt": [
                call["input_prompt"] for call in self.call_list[0]["calls"]
            ]
        }
        for guardrail_call in self.call_list:
            dataframe[guardrail_call["guardrail_name"] + ".safe"] = [
                call["outputs"]["safe"] for call in guardrail_call["calls"]
            ]
            dataframe[guardrail_call["guardrail_name"] + ".prediction_correctness"] = [
                call["score"]["correct"] for call in guardrail_call["calls"]
            ]
        return pd.DataFrame(dataframe)
