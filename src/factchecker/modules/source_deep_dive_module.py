"""Source deep dive module for generating site-specific follow-up queries."""

import dspy
from src.factchecker.signatures.source_deep_dive import SourceDeepDive


class SourceDeepDiveModule(dspy.Module):
    """Module that analyzes initial evidence and generates site-specific queries.

    This module enables multi-hop reasoning by:
    1. Analyzing initial evidence to identify promising authoritative sources
    2. Determining what information gaps exist
    3. Generating 1-3 targeted site-specific queries using "site:" operator

    Example: If initial evidence shows "PSEG Foundation gave $100K to TESU" but
    doesn't explain what programs TESU offers, this module will generate
    "site:tesu.edu scholarship programs" to discover program-level details.

    This is stage 2.25 of the fact-checking pipeline (after initial evidence
    retrieval, before evidence quality assessment).
    """

    def __init__(self):
        """Initialize the source deep dive module."""
        super().__init__()
        self.deep_dive = dspy.ChainOfThought(SourceDeepDive)

    def forward(self, statement: str, evidence: str) -> dspy.Prediction:
        """Generate site-specific queries for deeper investigation.

        Args:
            statement: The statement being fact-checked.
            evidence: Initial evidence gathered from web sources.

        Returns:
            dspy.Prediction with:
                - targeted_site_queries: List of 1-3 site-specific queries (or empty list)
                - reasoning: Explanation of source selection and information gaps
        """
        result = self.deep_dive(statement=statement, evidence=evidence)

        return dspy.Prediction(
            targeted_site_queries=result.targeted_site_queries,
            reasoning=result.reasoning,
        )
