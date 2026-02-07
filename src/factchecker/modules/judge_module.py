"""Simple judge module - barebones fact checker without research."""

import dspy
from typing import Optional
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
        topic: Optional[str] = None,
        url: Optional[str] = None,
        date_generated: Optional[str] = None
    ) -> dspy.Prediction:
        """Evaluate a statement for factual correctness.

        Args:
            statement: The statement to evaluate.
            topic: Optional topic/category metadata.
            url: Optional source URL metadata.
            date_generated: Optional date metadata.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
        """
        # Construct context-enriched statement if metadata is available
        context_parts = []
        if topic:
            context_parts.append(f"Topic: {topic}")
        if url:
            context_parts.append(f"Source: {url}")
        if date_generated:
            context_parts.append(f"Date: {date_generated}")

        if context_parts:
            context_string = "Context: " + ", ".join(context_parts) + f"\n\nStatement: {statement}"
        else:
            context_string = statement

        result = self.judge(statement=context_string)

        return dspy.Prediction(
            statement=statement,
            overall_verdict=result.verdict,
            confidence=result.confidence,
            reasoning=result.reasoning,
        )
