"""Simple judge module - barebones fact checker without research."""

import dspy
from typing import Optional, List
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

    def forward(self, statement: str, urls: Optional[List[str]] = None) -> dspy.Prediction:
        """Evaluate a statement for factual correctness.

        Args:
            statement: The statement to evaluate.
            urls: Optional list of URLs to scrape for evidence context.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
        """
        # Gather evidence from URLs if provided
        evidence_context = ""
        if urls:
            evidence_parts = []
            for url in urls:
                scraped = self.firecrawl_service.scrape(url)
                if scraped.success:
                    evidence_parts.append(f"Source: {url}\n{scraped.markdown}\n")
                else:
                    evidence_parts.append(f"Source: {url}\n[Failed to scrape: {scraped.error}]\n")
            evidence_context = "\n---\n".join(evidence_parts)

        result = self.judge(statement=statement, evidence_context=evidence_context)

        return dspy.Prediction(
            statement=statement,
            overall_verdict=result.verdict,
            confidence=result.confidence,
            reasoning=result.reasoning,
        )
