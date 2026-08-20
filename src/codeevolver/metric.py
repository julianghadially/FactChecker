"""Metric function for CodeEvolver optimization.

Inlines the scoring logic from gepa_metric, adapted for the CodeEvolver
metric contract: the metric is called with exactly two kwargs, `output`
(the pipeline prediction, untouched) and `example` (the full dataset row
wrapped so both `example["label"]` and `example.label` work).
"""

from src.evaluation.data_loader import FacToolLabelSchema


def asa_metric(output, example):
    """Score a single prediction against a ground-truth label.

    Handles both dict-style and object-style pipeline outputs. The gold
    label is read off the dataset row as `example["label"]`.
    """
    pred_label = (
        output.get("overall_verdict")
        if isinstance(output, dict)
        else getattr(output, "overall_verdict", None)
    )
    pred_label = FacToolLabelSchema.normalize_prediction(pred_label)
    gold_label = FacToolLabelSchema.normalize_ground_truth(example["label"])

    if pred_label == gold_label:
        return 1.0
    if pred_label == "UNKNOWN":
        return 0.5
    return 0.0


metric = asa_metric
