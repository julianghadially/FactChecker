"""Research module for generating fact-checking search queries."""

import dspy
from src.factchecker.signatures.research_strategy import ResearchStrategy


class ResearchModule(dspy.Module):
    """Generate search queries for fact-checking a statement.

    Uses DSPy to intelligently generate 2-3 targeted search queries
    that can help verify the factual claims in a statement. This is
    the first step in a research-enhanced fact-checking pipeline.

    Currently returns a placeholder evidence_summary (concatenated queries).
    Future enhancements will integrate SerperService and FirecrawlService
    to actually fetch and summarize evidence.
    """

    def __init__(self):
        """Initialize the research module."""
        super().__init__()
        self.researcher = dspy.ChainOfThought(ResearchStrategy)

    def forward(self, statement: str, topic: str = "") -> dspy.Prediction:
        """Generate search queries for fact-checking a statement.

        Args:
            statement: The statement to fact-check.
            topic: Optional topic/domain context for the statement.

        Returns:
            dspy.Prediction with:
                - queries: List of 2-3 search queries (list[str])
                - evidence_summary: Placeholder summary (str)
                - reasoning: Explanation of research strategy (str)
        """
        # Generate search queries using DSPy
        result = self.researcher(statement=statement, topic=topic)

        # Create placeholder evidence_summary
        # Future: Replace with actual SerperService + FirecrawlService integration
        evidence_summary = f"Planned search queries: {', '.join(result.search_queries)}"

        return dspy.Prediction(
            queries=result.search_queries,
            evidence_summary=evidence_summary,
            reasoning=result.reasoning
        )
