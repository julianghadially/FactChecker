"""Claim type analyzer module for classifying claims and generating search strategies."""

import dspy
from src.factchecker.signatures.claim_type_analyzer import ClaimTypeAnalyzer


class ClaimTypeAnalyzerModule(dspy.Module):
    """Module that analyzes claim type and suggests search strategy.

    Takes a statement as input and uses an LLM to:
    1. Classify the claim into one or more categories
    2. Generate targeted search strategy recommendations

    This enables the pipeline to adapt search queries based on claim
    characteristics, improving evidence retrieval for specific domains
    like corporate financial data or technical specifications.

    This is the first stage (Stage 0) of the evidence-aware fact-checking pipeline.
    """

    def __init__(self):
        """Initialize the claim type analyzer module."""
        super().__init__()
        self.analyzer = dspy.ChainOfThought(ClaimTypeAnalyzer)

    def forward(self, statement: str) -> dspy.Prediction:
        """Analyze claim type and generate search strategy.

        Args:
            statement: The claim to analyze.

        Returns:
            dspy.Prediction with:
                - claim_types: List of applicable claim type strings
                - search_strategy: Search strategy recommendations
                - reasoning: Explanation of the analysis
        """
        result = self.analyzer(statement=statement)

        return dspy.Prediction(
            claim_types=result.claim_types,
            search_strategy=result.search_strategy,
            reasoning=result.reasoning,
        )
