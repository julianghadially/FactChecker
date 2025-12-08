"""Metrics calculation for fact-checking evaluation."""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .data_loader import LabelSchema


@dataclass
class EvaluationMetrics:
    """Metrics from evaluation run."""

    accuracy: float
    total_examples: int
    valid_examples: int
    error_count: int
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    confusion_matrix: dict[str, dict[str, int]]


def calculate_metrics(
    predictions: list[str],
    ground_truth: list[str],
    schema: "LabelSchema"
) -> EvaluationMetrics:
    """Calculate evaluation metrics comparing predictions to ground truth.

    Args:
        predictions: List of predicted labels (raw from model).
        ground_truth: List of true labels from dataset.
        schema: Label schema for normalization.

    Returns:
        EvaluationMetrics with accuracy and per-class precision/recall.
    """
    # Normalize predictions and ground truth using schema
    normalized_preds = [schema.normalize_prediction(p) for p in predictions]
    normalized_truth = [schema.normalize_ground_truth(g) for g in ground_truth]

    # Count errors
    error_count = sum(1 for p in normalized_preds if p == "ERROR")

    # Filter out errors for metric calculation
    valid_pairs = [
        (p, g) for p, g in zip(normalized_preds, normalized_truth)
        if p != "ERROR"
    ]

    labels = schema.get_labels()

    if not valid_pairs:
        return EvaluationMetrics(
            accuracy=0.0,
            total_examples=len(ground_truth),
            valid_examples=0,
            error_count=error_count,
            per_class_precision={label: 0.0 for label in labels},
            per_class_recall={label: 0.0 for label in labels},
            confusion_matrix={label: {l: 0 for l in labels} for label in labels}
        )

    valid_preds, valid_truth = zip(*valid_pairs)

    # Calculate accuracy
    correct = sum(p == g for p, g in zip(valid_preds, valid_truth))
    accuracy = correct / len(valid_truth)

    # Build confusion matrix
    confusion = {true_label: {pred_label: 0 for pred_label in labels} for true_label in labels}
    for pred, true in zip(valid_preds, valid_truth):
        if true in confusion and pred in labels:
            confusion[true][pred] += 1

    # Calculate per-class precision and recall
    precisions = {}
    recalls = {}

    for label in labels:
        # True positives: predicted label correctly
        tp = confusion[label][label]

        # False positives: predicted label but wrong
        fp = sum(confusion[other][label] for other in labels if other != label)

        # False negatives: was label but predicted something else
        fn = sum(confusion[label][other] for other in labels if other != label)

        precisions[label] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recalls[label] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return EvaluationMetrics(
        accuracy=accuracy,
        total_examples=len(ground_truth),
        valid_examples=len(valid_pairs),
        error_count=error_count,
        per_class_precision=precisions,
        per_class_recall=recalls,
        confusion_matrix=confusion
    )


def get_f1(metrics: EvaluationMetrics, label: str) -> float:
    """Calculate F1 score for a specific class.

    Args:
        metrics: EvaluationMetrics object.
        label: Class label to calculate F1 for.

    Returns:
        F1 score as float.
    """
    precision = metrics.per_class_precision.get(label, 0.0)
    recall = metrics.per_class_recall.get(label, 0.0)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def print_metrics(metrics: EvaluationMetrics, name: str = "Model") -> None:
    """Print metrics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"{name} Results")
    print(f"{'='*60}")
    print(f"Total examples:  {metrics.total_examples}")
    print(f"Valid examples:  {metrics.valid_examples}")
    print(f"Errors:          {metrics.error_count}")
    print(f"Accuracy:        {metrics.accuracy:.1%}")
    print()
    
    # Highlight key metrics for FacTool evaluation
    print("=" * 60)
    print("KEY METRICS (Focus Areas)")
    print("=" * 60)
    
    # Precision of SUPPORTED
    if "SUPPORTED" in metrics.per_class_precision:
        supported_precision = metrics.per_class_precision["SUPPORTED"]
        print(f"Precision of SUPPORTED: {supported_precision:.1%} ⭐")
    
    # Precision of REFUTED
    if "REFUTED" in metrics.per_class_precision:
        refuted_precision = metrics.per_class_precision["REFUTED"]
        print(f"Precision of REFUTED:   {refuted_precision:.1%} ⭐")
    
    # Recall of REFUTED
    if "REFUTED" in metrics.per_class_recall:
        refuted_recall = metrics.per_class_recall["REFUTED"]
        print(f"Recall of REFUTED:      {refuted_recall:.1%} ⭐")
    
    print()
    print("=" * 60)
    print("ALL METRICS")
    print("=" * 60)
    print("Per-class Precision:")
    for label, precision in metrics.per_class_precision.items():
        marker = " ⭐" if label in ["SUPPORTED", "REFUTED"] else ""
        print(f"  {label}: {precision:.1%}{marker}")
    print()
    print("Per-class Recall:")
    for label, recall in metrics.per_class_recall.items():
        marker = " ⭐" if label == "REFUTED" else ""
        print(f"  {label}: {recall:.1%}{marker}")
    print()
    print("Confusion Matrix (rows=truth, cols=pred):")
    labels = list(metrics.confusion_matrix.keys())
    header = "".ljust(20) + "".join(l[:12].ljust(14) for l in labels)
    print(header)
    for true_label in labels:
        row = true_label.ljust(20)
        for pred_label in labels:
            row += str(metrics.confusion_matrix[true_label][pred_label]).ljust(14)
        print(row)
    print(f"{'='*60}")
