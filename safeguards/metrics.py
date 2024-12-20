from typing import Optional

import numpy as np
import weave


class AccuracyMetric(weave.Scorer):
    """
    A class to compute and summarize accuracy-related metrics for model outputs.

    This class extends the `weave.Scorer` and provides operations to score
    individual predictions and summarize the results across multiple predictions.
    It calculates the accuracy, precision, recall, and F1 score based on the
    comparison between predicted outputs and true labels.
    """

    @weave.op()
    def score(self, output: dict, label: int):
        """
        Evaluate the correctness of a single prediction.

        This method compares a model's predicted output with the true label
        to determine if the prediction is correct. It checks if the 'safe'
        field in the output dictionary, when converted to an integer, matches
        the provided label.

        The scorer assumes that the dataset labels are 0 for safe and 1 for unsafe.

        Args:
            output (dict): A dictionary containing the model's prediction,
                specifically the 'safe' key which holds the predicted value.
            label (int): The true label against which the prediction is compared.

        Returns:
            dict: A dictionary with a single key 'correct', which is True if the
          prediction matches the label, otherwise False.
        """
        return {"correct": label != int(output["safe"])}

    @weave.op()
    def summarize(self, score_rows: list) -> Optional[dict]:
        """
        Summarize the accuracy-related metrics from a list of prediction scores.

        This method processes a list of score dictionaries, each containing a
        'correct' key indicating whether a prediction was correct. It calculates
        several metrics: accuracy, precision, recall, and F1 score, based on the
        number of true positives, false positives, and false negatives.

        Args:
            score_rows (list): A list of dictionaries, each with a 'correct' key
              indicating the correctness of individual predictions.

        Returns:
            Optional[dict]: A dictionary containing the calculated metrics:
                'accuracy', 'precision', 'recall', and 'f1_score'. If no valid data
                is present, all metrics default to 0.
        """
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
