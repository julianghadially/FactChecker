"""Simple judge module - barebones fact checker without research."""

import dspy
from src.factchecker.simple.signatures.judge import Judge
from src.factchecker.modules.url_prefetch_module import UrlPrefetchModule


class JudgeModule(dspy.Module):
    """Barebones fact checker that judges statements without research.

    Takes a statement as input and outputs a verdict directly using LLM knowledge.
    No claim extraction, no web search, no evidence gathering.

    This serves as a simpler/faster alternative to the full FactCheckerPipeline
    for cases where external research is not needed or desired.

    Optionally accepts URLs to pre-fetch and enrich context before judging.
    """

    def __init__(self):
        """Initialize the simple judge module."""
        super().__init__()
        self.url_prefetch = UrlPrefetchModule()
        self.judge = dspy.ChainOfThought(Judge)

    def forward(self, statement: str, urls: list[str] | None = None) -> dspy.Prediction:
        """Evaluate a statement for factual correctness.

        Args:
            statement: The statement to evaluate.
            urls: Optional list of URLs to scrape for additional evidence context.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
                - urls_processed: Number of URLs successfully processed (if URLs provided)
        """
        # Pre-fetch and enrich context from URLs if provided
        prefetch_result = self.url_prefetch(statement=statement, urls=urls)
        enriched_context = prefetch_result.enriched_context

        # Judge using enriched context (or original statement if no URLs)
        result = self.judge(statement=enriched_context)

        return dspy.Prediction(
            statement=statement,
            overall_verdict=result.verdict,
            confidence=result.confidence,
            reasoning=result.reasoning,
            urls_processed=prefetch_result.urls_processed,
        )
