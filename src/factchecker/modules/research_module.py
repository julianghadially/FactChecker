"""Research module for generating search queries and evidence summaries."""

import dspy
from src.factchecker.signatures.research_signature import ResearchSignature


class ResearchModule(dspy.Module):
    """Research module that generates search queries for fact-checking.

    Uses an LLM to generate 2-3 targeted search queries based on a statement
    and topic. In the current phase, creates a placeholder evidence_summary.
    Future versions will integrate with SERPER for search and Firecrawl for
    page scraping to build comprehensive evidence.
    """

    def __init__(self):
        """Initialize the research module."""
        super().__init__()
        self.query_generator = dspy.ChainOfThought(ResearchSignature)

    def forward(self, statement: str, topic: str) -> dspy.Prediction:
        """Generate search queries for a statement.

        Args:
            statement: The statement to research.
            topic: The topic/domain of the statement.

        Returns:
            dspy.Prediction with:
                - queries: List of 2-3 search query strings
                - reasoning: Explanation of query generation strategy
                - evidence_summary: Placeholder summary (currently concatenates queries)
        """
        # Generate search queries using the LLM
        result = self.query_generator(statement=statement, topic=topic)

        # Create placeholder evidence summary
        # Future: This will be replaced with actual search results and scraped content
        evidence_summary = (
            f"Generated {len(result.search_queries)} search queries for research: "
            + "; ".join(result.search_queries)
        )

        return dspy.Prediction(
            queries=result.search_queries,
            reasoning=result.reasoning,
            evidence_summary=evidence_summary
        )
