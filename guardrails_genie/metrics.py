from typing import Optional

import numpy as np
import weave


class AccuracyMetric(weave.Scorer):
    @weave.op()
    def score(self, output: dict, label: int):
        return {"correct": bool(label) == output["safe"]}

    @weave.op()
    def summarize(self, score_rows: list) -> Optional[dict]:
        valid_data = [
            x.get("correct") for x in score_rows if x.get("correct") is not None
        ]
        count_true = list(valid_data).count(True)
        int_data = [int(x) for x in valid_data]

        true_positives = count_true
        false_positives = len(valid_data) - count_true
        false_negatives = len(score_rows) - len(valid_data)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1_score = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "accuracy": float(np.mean(int_data) if int_data else 0),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }
