"""Evaluation system for comparing fact-checker against baseline."""

from .data_loader import load_dataset, HoverExample
from .metrics import calculate_metrics, EvaluationMetrics
from .evaluate import run_evaluation

__all__ = [
    "load_dataset",
    "HoverExample",
    "calculate_metrics",
    "EvaluationMetrics",
    "run_evaluation",
]
