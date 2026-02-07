"""Search query generator module for creating targeted web search queries."""

import dspy
from src.factchecker.signatures.search_query_generator import SearchQueryGenerator


class SearchQueryGeneratorModule(dspy.Module):
    """Module that generates targeted search queries using a two-phase strategy.

    Takes a statement as input and uses an LLM to generate two types of queries:
    1. Primary source queries (1-2): Site-specific queries targeting authoritative sources
    2. General queries (1-2): Broader queries for context and verification

    This is the first stage of the evidence-aware fact-checking pipeline.
    """

    def __init__(self):
        """Initialize the search query generator module."""
        super().__init__()
        self.generator = dspy.ChainOfThought(SearchQueryGenerator)

    def forward(self, statement: str) -> dspy.Prediction:
        """Generate search queries for a statement using a two-phase strategy.

        Args:
            statement: The statement to generate queries for.

        Returns:
            dspy.Prediction with:
                - primary_source_queries: List of 1-2 site-specific queries targeting authoritative sources
                - general_queries: List of 1-2 broader queries for context
                - reasoning: Explanation of the two-phase query strategy
        """
        result = self.generator(statement=statement)

        return dspy.Prediction(
            primary_source_queries=result.primary_source_queries,
            general_queries=result.general_queries,
            reasoning=result.reasoning,
        )
