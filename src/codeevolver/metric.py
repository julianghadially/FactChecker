"""Metric function for CodeEvolver GEPA optimization.

Re-exports the same metric used in gepa_optimize.py for consistency.
"""

from src.optimizer.gepa_optimize import gepa_metric


def asa_metric(output, label):
    """Wrapper around gepa_metric for the ASA interface."""
    return gepa_metric(pred=output, gold=label)


metric = asa_metric
