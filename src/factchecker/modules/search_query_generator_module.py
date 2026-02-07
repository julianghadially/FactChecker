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

    def forward(
        self,
        statement: str,
        claim_types: list[str] = None,
        search_strategy: str = None,
    ) -> dspy.Prediction:
        """Generate search queries for a statement.

        Args:
            statement: The statement to generate queries for.
            claim_types: Optional list of claim types from ClaimTypeAnalyzer.
            search_strategy: Optional search strategy recommendations.

        Returns:
            dspy.Prediction with:
                - queries: List of 1-3 search query strings
                - reasoning: Explanation of the query strategy
        """
        # Use defaults if not provided for backward compatibility
        if claim_types is None:
            claim_types = []
        if search_strategy is None:
            search_strategy = ""

        result = self.generator(
            statement=statement,
            claim_types=claim_types,
            search_strategy=search_strategy,
        )

        return dspy.Prediction(
            queries=result.queries,
            reasoning=result.reasoning,
        )
