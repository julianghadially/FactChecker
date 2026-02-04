"""Simple judge module - barebones fact checker without research."""

import dspy
from typing import Optional
from src.factchecker.simple.signatures.judge import Judge
from src.services.firecrawl_service import FirecrawlService


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
        self.firecrawl_service = FirecrawlService()

    def forward(self, statement: str, url: Optional[str] = None) -> dspy.Prediction:
        """Evaluate a statement for factual correctness.

        Args:
            statement: The statement to evaluate.
            url: Optional URL(s) to scrape for evidence. Can be a single URL or
                 comma-separated URLs. If provided, content will be scraped and
                 used as evidence for verification.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
        """
        evidence = ""

        # If URL(s) provided, attempt to scrape evidence
        if url:
            urls = [u.strip() for u in url.split(',')]
            evidence_parts = []

            for single_url in urls:
                try:
                    scraped = self.firecrawl_service.scrape(single_url)
                    if scraped.success and scraped.markdown:
                        evidence_parts.append(f"--- Evidence from {single_url} ---\n{scraped.markdown}\n")
                    else:
                        # Log failure but continue - we'll fall back to knowledge-only judgment
                        error_msg = scraped.error if scraped.error else "Unknown error"
                        print(f"Warning: Failed to scrape {single_url}: {error_msg}")
                except Exception as e:
                    # Catch any unexpected errors and continue gracefully
                    print(f"Warning: Exception while scraping {single_url}: {str(e)}")

            if evidence_parts:
                evidence = "\n".join(evidence_parts)

        # Call the judge with statement and evidence (empty string if no evidence)
        result = self.judge(statement=statement, evidence=evidence)

        return dspy.Prediction(
            statement=statement,
            overall_verdict=result.verdict,
            confidence=result.confidence,
            reasoning=result.reasoning,
        )
