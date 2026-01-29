"""Metric function for CodeEvolver GEPA optimization.

Re-exports the same metric used in gepa_optimize.py for consistency.
"""

from src.optimizer.gepa_optimize import gepa_metric

# CodeEvolver expects a function named 'metric'
metric = gepa_metric
