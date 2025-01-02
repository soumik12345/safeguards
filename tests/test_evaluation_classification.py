from safeguards.meta_evaluation import EvaluationClassifier


def test_evaluation_classification():
    parser = EvaluationClassifier(
        project="geekyrakshit/guardrails-genie",
        call_id="0193bc3f-cb02-7271-89bd-e7fdccfc4edc",
    )
    parser.register_predict_and_score_calls(
        max_predict_and_score_calls=2, save_filepath="evaluation.json"
    )
    assert len(parser.predict_and_score_calls) == 2
