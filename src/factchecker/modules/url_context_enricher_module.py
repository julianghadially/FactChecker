"""URL Context Enricher module that wraps JudgeModule with URL-based evidence."""

import dspy
from typing import Optional
from src.factchecker.simple.modules.judge_module import JudgeModule
from src.services.firecrawl_service import FirecrawlService


class UrlContextEnricherModule(dspy.Module):
    """Preprocessing module that enriches statements with evidence from URLs.

    This module wraps JudgeModule and automatically scrapes provided URLs to extract
    key facts before passing the enriched context to the judge. This allows the judge
    to make evidence-based verdicts instead of relying solely on LLM knowledge.

    Attributes:
        judge: The underlying JudgeModule for statement evaluation.
        firecrawl: Service for scraping web pages.
        max_urls: Maximum number of URLs to process (default: 2).
        max_chars_per_url: Maximum characters to extract per URL (default: 1000).
    """

    def __init__(
        self,
        judge: Optional[JudgeModule] = None,
        max_urls: int = 2,
        max_chars_per_url: int = 1000
    ):
        """Initialize the URL context enricher module.

        Args:
            judge: JudgeModule instance to wrap. If None, creates a new one.
            max_urls: Maximum number of URLs to scrape (default: 2).
            max_chars_per_url: Maximum characters to extract per URL (default: 1000).
        """
        super().__init__()
        self.judge = judge if judge is not None else JudgeModule()
        self.firecrawl = FirecrawlService()
        self.max_urls = max_urls
        self.max_chars_per_url = max_chars_per_url

    def _extract_facts_from_content(self, markdown: str, max_chars: int) -> str:
        """Extract key facts from scraped content.

        Args:
            markdown: The scraped markdown content.
            max_chars: Maximum characters to extract.

        Returns:
            Truncated content with key facts.
        """
        # Simple truncation for now - could be enhanced with LLM summarization
        if len(markdown) > max_chars:
            return markdown[:max_chars] + "..."
        return markdown

    def _enrich_statement_with_urls(
        self,
        statement: str,
        urls: list[str]
    ) -> str:
        """Scrape URLs and enrich the statement with extracted context.

        Args:
            statement: The original statement to evaluate.
            urls: List of URLs to scrape for context.

        Returns:
            Enriched statement with context prepended.
        """
        if not urls:
            return statement

        # Limit to max_urls
        urls_to_scrape = urls[:self.max_urls]

        context_parts = []
        for url in urls_to_scrape:
            try:
                scraped = self.firecrawl.scrape(
                    url,
                    max_length=self.max_chars_per_url * 2  # Give some buffer for truncation
                )

                if scraped.success and scraped.markdown:
                    facts = self._extract_facts_from_content(
                        scraped.markdown,
                        self.max_chars_per_url
                    )
                    context_parts.append(f"[{url}]: {facts}")
                else:
                    # Log error but continue with other URLs
                    error_msg = scraped.error or "Failed to scrape"
                    context_parts.append(f"[{url}]: Error - {error_msg}")
            except Exception as e:
                # Catch any unexpected errors and continue
                context_parts.append(f"[{url}]: Error - {str(e)}")

        # Build enriched statement
        if context_parts:
            context_section = "Context from provided sources:\n" + "\n".join(context_parts)
            enriched = f"{context_section}\n\nStatement to evaluate: {statement}"
            return enriched

        return statement

    def forward(
        self,
        statement: str,
        url: Optional[str] = None,
        urls: Optional[list[str]] = None
    ) -> dspy.Prediction:
        """Evaluate a statement with optional URL context enrichment.

        Args:
            statement: The statement to evaluate.
            url: Single URL to scrape for context (optional).
            urls: List of URLs to scrape for context (optional).

        Returns:
            dspy.Prediction with:
                - statement: The original input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
        """
        # Collect all URLs
        all_urls = []
        if url:
            all_urls.append(url)
        if urls:
            all_urls.extend(urls)

        # Enrich statement with URL context if URLs provided
        if all_urls:
            enriched_statement = self._enrich_statement_with_urls(statement, all_urls)
        else:
            enriched_statement = statement

        # Pass enriched statement to judge
        result = self.judge.forward(enriched_statement)

        # Return result with original statement (not enriched version)
        return dspy.Prediction(
            statement=statement,
            overall_verdict=result.overall_verdict,
            confidence=result.confidence,
            reasoning=result.reasoning,
        )
