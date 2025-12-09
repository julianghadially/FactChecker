"""Claim extractor module for extracting factual claims from statements."""
import time
import dspy
from src.factchecker.signatures.claim_extractor import ClaimExtractor


class ClaimExtractorModule(dspy.Module):
    """Module that extracts individual factual claims from a statement.

    Uses chain-of-thought reasoning to identify and separate distinct
    factual claims that can be independently verified.
    """

    def __init__(self):
        """Initialize the claim extractor module."""
        super().__init__()
        self.extractor = dspy.ChainOfThought(ClaimExtractor)

    def forward(self, statement: str) -> dspy.Prediction:
        """Extract claims from a statement.

        Args:
            statement: The input statement to analyze.

        Returns:
            List of distinct factual claims extracted from the statement.
        """
        start_time = time.time()
        result = self.extractor(statement=statement)
        time_taken = time.time() - start_time
        if time_taken > 0.5:
            print(f"Claim extractor time. Statement: {statement}. \nTime: {time_taken:.2f} seconds")
        return dspy.Prediction(claims=result.claims)