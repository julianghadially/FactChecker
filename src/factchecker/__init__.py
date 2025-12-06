"""FactChecker - DSPy-based multi-hop fact verification system."""

from .modules.fact_checker_pipeline import FactCheckerPipeline
from .models.data_types import FactCheckResult, JudgmentResult, AggregationResult

__all__ = [
    "FactCheckerPipeline",
    "FactCheckResult",
    "JudgmentResult",
    "AggregationResult",
]
