"""Metric function for CodeEvolver GEPA optimization.

Inlines the scoring logic from gepa_metric, adapted for the ASA interface
where `output` is the pipeline prediction and `label` is a bare string.
"""

from src.evaluation.data_loader import FacToolLabelSchema


def asa_metric(output, label):
    """Score a single prediction against a ground-truth label.

    Handles both dict-style and object-style pipeline outputs.
    """
    pred_label = (
        output.get("overall_verdict")
        if isinstance(output, dict)
        else getattr(output, "overall_verdict", None)
    )
    pred_label = FacToolLabelSchema.normalize_prediction(pred_label)
    gold_label = FacToolLabelSchema.normalize_ground_truth(label)

    if pred_label == gold_label:
        return 1.0
    if pred_label == "UNKNOWN":
        return 0.5
    return 0.0


metric = asa_metric
