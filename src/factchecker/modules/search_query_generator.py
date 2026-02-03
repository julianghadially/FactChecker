"""Search Query Generator module for creating initial search queries from claims."""

import dspy
from src.factchecker.signatures.search_query_generator import SearchQueryGenerator


class SearchQueryGeneratorModule(dspy.Module):
    """Generate optimized initial search queries from claims.

    Uses DSPy ChainOfThought to analyze claims and extract key elements
    (entities, metrics, time periods, action verbs) to construct targeted
    search queries optimized for web search engines.

    This ensures every claim starts with relevant web evidence rather than
    relying solely on the LLM's internal knowledge.
    """

    def __init__(self):
        """Initialize the Search Query Generator module."""
        super().__init__()
        self.generator = dspy.ChainOfThought(SearchQueryGenerator)

    def forward(self, claim: str) -> dspy.Prediction:
        """Generate an optimized search query from a claim.

        Args:
            claim: The factual claim to generate a search query for.

        Returns:
            dspy.Prediction with reasoning and search_query fields.
        """
        return self.generator(claim=claim)
