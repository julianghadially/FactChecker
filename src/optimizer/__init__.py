"""Optimizer module for GEPA-based optimization of FactChecker pipeline."""

from .gepa_optimize import run_optimization, gepa_metric, load_dspy_examples

__all__ = ["run_optimization", "gepa_metric", "load_dspy_examples"]
