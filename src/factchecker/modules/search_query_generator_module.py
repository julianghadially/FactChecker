"""Search query generator module for creating targeted web search queries."""

import dspy
from src.factchecker.signatures.search_query_generator import SearchQueryGenerator


class SearchQueryGeneratorModule(dspy.Module):
    """Module that generates targeted search queries for fact-checking.

    Takes a statement as input and uses an LLM to generate 1-3 specific,
    diverse search queries that will help gather evidence to verify or refute
    the claims in the statement.

    This is the first stage of the evidence-aware fact-checking pipeline.
    """

    def __init__(self):
        """Initialize the search query generator module."""
        super().__init__()
        self.generator = dspy.ChainOfThought(SearchQueryGenerator)

    def forward(self, statement: str) -> dspy.Prediction:
        """Generate search queries for a statement.

        Args:
            statement: The statement to generate queries for.

        Returns:
            dspy.Prediction with:
                - queries: List of 1-3 search query strings
                - reasoning: Explanation of the query strategy
        """
        result = self.generator(statement=statement)

        return dspy.Prediction(
            queries=result.queries,
            reasoning=result.reasoning,
        )
