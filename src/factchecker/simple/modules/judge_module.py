"""Simple judge module - barebones fact checker without research."""

import dspy
from typing import Optional
from src.factchecker.simple.signatures.judge import Judge


class JudgeModule(dspy.Module):
    """Barebones fact checker that judges statements without research.

    Takes a statement as input and outputs a verdict directly using LLM knowledge.
    No claim extraction, no web search, no evidence gathering.

    This serves as a simpler/faster alternative to the full FactCheckerPipeline
    for cases where external research is not needed or desired.
    """

    def __init__(self, knowledge_cutoff_date: str = "20240401"):
        """Initialize the simple judge module.

        Args:
            knowledge_cutoff_date: The LLM's knowledge cutoff date in YYYYMMDD format.
                                   Defaults to "20240401" (April 1, 2024).
        """
        super().__init__()
        self.judge = dspy.ChainOfThought(Judge)
        self.knowledge_cutoff_date = knowledge_cutoff_date

    def forward(self, statement: str, statement_date: Optional[str] = None) -> dspy.Prediction:
        """Evaluate a statement for factual correctness.

        Args:
            statement: The statement to evaluate.
            statement_date: Optional date the statement was made/generated (format: YYYYMMDD).
                           Used to assess temporal context.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
        """
        result = self.judge(
            statement=statement,
            statement_date=statement_date,
            knowledge_cutoff_date=self.knowledge_cutoff_date,
        )

        return dspy.Prediction(
            statement=statement,
            overall_verdict=result.verdict,
            confidence=result.confidence,
            reasoning=result.reasoning,
        )
