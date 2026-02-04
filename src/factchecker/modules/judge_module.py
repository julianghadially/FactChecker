"""Simple judge module - barebones fact checker without research."""

import dspy
from src.factchecker.signatures.judge import Judge


class JudgeModule(dspy.Module):
    """Barebones fact checker that judges statements without research.

    Takes a statement as input and outputs a verdict directly using LLM knowledge.
    No claim extraction, no web search, no evidence gathering.

    This serves as a simpler/faster alternative to the full FactCheckerPipeline
    for cases where external research is not needed or desired.
    """

    def __init__(self):
        """Initialize the simple judge module."""
        super().__init__()
        self.judge = dspy.ChainOfThought(Judge)

    def forward(
        self,
        statement: str,
        topic: str = "",
        date: str = "",
        source_urls: str = ""
    ) -> dspy.Prediction:
        """Evaluate a statement for factual correctness.

        Args:
            statement: The statement to evaluate.
            topic: Optional context about the topic or domain.
            date: Optional context about when the statement was generated (YYYYMMDD format).
            source_urls: Optional comma-separated URLs providing relevant context.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
        """
        result = self.judge(
            statement=statement,
            topic=topic,
            date=date,
            source_urls=source_urls
        )

        return dspy.Prediction(
            statement=statement,
            overall_verdict=result.verdict,
            confidence=result.confidence,
            reasoning=result.reasoning,
        )
