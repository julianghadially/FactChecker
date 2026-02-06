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
        url: str = "",
        date_generated: str = ""
    ) -> dspy.Prediction:
        """Evaluate a statement for factual correctness.

        Args:
            statement: The statement to evaluate.
            topic: Optional topic/domain context (default: "").
            url: Optional reference URL (default: "").
            date_generated: Optional creation date (default: "").

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
            url=url,
            date_generated=date_generated
        )

        return dspy.Prediction(
            statement=statement,
            overall_verdict=result.verdict,
            confidence=result.confidence,
            reasoning=result.reasoning,
        )
