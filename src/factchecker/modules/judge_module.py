"""Simple judge module - barebones fact checker without research."""

from typing import Optional
import dspy
from src.factchecker.signatures.judge import Judge
from src.factchecker.modules.context_enrichment_module import ContextEnrichmentModule


class JudgeModule(dspy.Module):
    """Barebones fact checker that judges statements without research.

    Takes a statement as input and outputs a verdict directly using LLM knowledge.
    No claim extraction, no web search, no evidence gathering.

    This serves as a simpler/faster alternative to the full FactCheckerPipeline
    for cases where external research is not needed or desired.

    Optionally accepts URLs to scrape for additional context to aid verification.
    """

    def __init__(self):
        """Initialize the simple judge module."""
        super().__init__()
        self.judge = dspy.ChainOfThought(Judge)
        self.context_enrichment = None  # Lazy initialization

    def forward(self, statement: str, url: Optional[str] = None) -> dspy.Prediction:
        """Evaluate a statement for factual correctness.

        Args:
            statement: The statement to evaluate.
            url: Optional comma-separated URLs to scrape for additional context.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
        """
        context = None

        # If URLs are provided, scrape them for context
        if url:
            if self.context_enrichment is None:
                self.context_enrichment = ContextEnrichmentModule()

            # Split by comma and scrape all URLs
            urls = [u.strip() for u in url.split(",") if u.strip()]
            if urls:
                context = self.context_enrichment.forward(urls)

        # Pass context to judge (will be None if no URLs provided)
        result = self.judge(statement=statement, context=context)

        return dspy.Prediction(
            statement=statement,
            overall_verdict=result.verdict,
            confidence=result.confidence,
            reasoning=result.reasoning,
        )
