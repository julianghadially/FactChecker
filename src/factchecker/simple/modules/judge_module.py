"""Simple judge module - barebones fact checker without research."""

import dspy
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

    def forward(self, statement: str, urls: list[str] | None = None) -> dspy.Prediction:
        """Evaluate a statement for factual correctness.

        Args:
            statement: The statement to evaluate.
            urls: Optional list of URLs to scrape for evidence context.
                  Only the first 3 URLs will be scraped for cost efficiency.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
        """
        # Scrape URLs for evidence context if provided
        evidence_context = ""
        if urls:
            # Limit to first 3 URLs for cost efficiency
            urls_to_scrape = urls[:3]
            scraped_contents = []

            for url in urls_to_scrape:
                scraped_page = self.firecrawl_service.scrape(url)
                if scraped_page.success and scraped_page.markdown:
                    scraped_contents.append(
                        f"Source: {url}\nContent: {scraped_page.markdown}\n\n"
                    )

            evidence_context = "".join(scraped_contents)

        result = self.judge(statement=statement, evidence_context=evidence_context)

        return dspy.Prediction(
            statement=statement,
            overall_verdict=result.verdict,
            confidence=result.confidence,
            reasoning=result.reasoning,
        )
