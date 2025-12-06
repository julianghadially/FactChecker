"""Claim extractor module for extracting factual claims from statements."""

import dspy
from ..signatures.claim_extractor import ClaimExtractor


class ClaimExtractorModule(dspy.Module):
    """Module that extracts individual factual claims from a statement.

    Uses chain-of-thought reasoning to identify and separate distinct
    factual claims that can be independently verified.
    """

    def __init__(self):
        """Initialize the claim extractor module."""
        super().__init__()
        self.extractor = dspy.ChainOfThought(ClaimExtractor)

    def forward(self, statement: str) -> list[str]:
        """Extract claims from a statement.

        Args:
            statement: The input statement to analyze.

        Returns:
            List of distinct factual claims extracted from the statement.
        """
        result = self.extractor(statement=statement)
        return result.claims
