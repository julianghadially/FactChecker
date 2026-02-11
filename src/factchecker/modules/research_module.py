"""Research module for generating search queries to verify statements."""

import dspy
from src.factchecker.signatures.research import Research


class ResearchModule(dspy.Module):
    """Research module that generates search queries for fact verification.

    Takes a statement and topic as input, uses chain-of-thought reasoning
    to generate 2-3 targeted search queries that would help verify the
    statement's factual claims.

    This is the first stage of the FactCheckerPipeline. In the current
    implementation, evidence gathering is a placeholder. Future versions
    will integrate SERPER/Firecrawl to actually execute searches.
    """

    def __init__(self):
        """Initialize the research module."""
        super().__init__()
        self.research = dspy.ChainOfThought(Research)

    def forward(self, statement: str, topic: str) -> dspy.Prediction:
        """Generate search queries for a statement.

        Args:
            statement: The statement to research.
            topic: The topic/domain of the statement.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - topic: The input topic
                - queries: List of 2-3 search queries
                - evidence_summary: Placeholder string (concatenated queries)
                - reasoning: Explanation of research strategy
        """
        result = self.research(statement=statement, topic=topic)

        # Placeholder: concatenate queries as evidence summary
        # Future: execute searches with SERPER and scrape with Firecrawl
        evidence_summary = "Research queries generated: " + "; ".join(result.search_queries)

        return dspy.Prediction(
            statement=statement,
            topic=topic,
            queries=result.search_queries,
            evidence_summary=evidence_summary,
            reasoning=result.reasoning
        )
