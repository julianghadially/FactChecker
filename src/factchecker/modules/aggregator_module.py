"""Aggregator module for combining claim verdicts into overall statement verdict."""

import dspy
from src.factchecker.signatures.aggregator import Aggregator
from src.factchecker.models.data_types import AggregationResult


class AggregatorModule(dspy.Module):
    """Aggregates individual claim verdicts into overall statement verdict.

    Applies priority logic:
    1. Any refuted claim -> CONTAINS_REFUTED_CLAIMS
    2. Any not_supported claim -> CONTAINS_UNSUPPORTED_CLAIMS
    3. All supported -> SUPPORTED
    """

    def __init__(self):
        """Initialize the aggregator module."""
        super().__init__()
        self.aggregator = dspy.ChainOfThought(Aggregator)

    def forward(
        self,
        original_statement: str,
        claim_verdicts: list[dict]
    ) -> dspy.Prediction:
        """Aggregate claim verdicts into overall verdict.

        Args:
            original_statement: The original statement being evaluated.
            claim_verdicts: List of dicts with 'claim', 'verdict', 'evidence_summary'.

        Returns:
            AggregationResult with overall verdict and confidence.
        """
        result = self.aggregator(
            original_statement=original_statement,
            claim_verdicts=claim_verdicts
        )

        return dspy.Prediction(
            verdict=result.overall_verdict,
            confidence=result.confidence,
            reasoning=result.reasoning,
            claim_details=claim_verdicts
        )
