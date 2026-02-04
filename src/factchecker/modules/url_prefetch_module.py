"""URL prefetch module for enriching context with evidence from provided URLs."""

import concurrent.futures
from typing import Optional

import dspy

from src.factchecker.signatures.context_enricher import ContextEnricher
from src.services.firecrawl_service import FirecrawlService, ScrapedPage


class UrlPrefetchModule(dspy.Module):
    """Context enrichment preprocessor that scrapes and summarizes evidence from URLs.

    This module accepts a statement and optional URLs, scrapes the URLs in parallel,
    and uses DSPy to extract relevant evidence that can help evaluate the statement.

    The enriched context combines the original statement with summarized evidence
    from each URL, which can then be used by downstream modules like JudgeModule.
    """

    def __init__(self, max_urls: int = 3, max_content_length: int = 10000):
        """Initialize the URL prefetch module.

        Args:
            max_urls: Maximum number of URLs to scrape (default: 3)
            max_content_length: Maximum characters per scraped page (default: 10000)
        """
        super().__init__()
        self.max_urls = max_urls
        self.max_content_length = max_content_length
        self.firecrawl_service = FirecrawlService()
        self.enricher = dspy.ChainOfThought(ContextEnricher)

    def _scrape_url(self, url: str) -> ScrapedPage:
        """Scrape a single URL using FirecrawlService.

        Args:
            url: URL to scrape

        Returns:
            ScrapedPage with markdown content or error information
        """
        return self.firecrawl_service.scrape(
            url=url,
            max_length=self.max_content_length
        )

    def _extract_evidence(
        self,
        statement: str,
        scraped_page: ScrapedPage
    ) -> Optional[str]:
        """Extract relevant evidence from a scraped page.

        Args:
            statement: The statement being evaluated
            scraped_page: The scraped page content

        Returns:
            Extracted evidence string with source attribution, or None if extraction fails
        """
        if not scraped_page.success or not scraped_page.markdown:
            return None

        try:
            result = self.enricher(
                statement=statement,
                page_content=scraped_page.markdown,
                source_url=scraped_page.url
            )
            return result.relevant_evidence
        except Exception as e:
            print(f"Warning: Failed to extract evidence from {scraped_page.url}: {e}")
            return None

    def forward(
        self,
        statement: str,
        urls: Optional[list[str]] = None
    ) -> dspy.Prediction:
        """Enrich context by scraping URLs and extracting relevant evidence.

        Args:
            statement: The statement to evaluate
            urls: Optional list of URLs to scrape for evidence

        Returns:
            dspy.Prediction with:
                - enriched_context: Statement with appended evidence from URLs
                - urls_processed: Number of URLs successfully processed
        """
        # If no URLs provided, return original statement
        if not urls:
            return dspy.Prediction(
                enriched_context=statement,
                urls_processed=0
            )

        # Limit to max_urls
        urls_to_scrape = urls[:self.max_urls]

        # Scrape URLs in parallel
        scraped_pages: list[ScrapedPage] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_urls) as executor:
            futures = [executor.submit(self._scrape_url, url) for url in urls_to_scrape]
            scraped_pages = [
                future.result()
                for future in concurrent.futures.as_completed(futures)
            ]

        # Extract evidence from each scraped page
        evidence_pieces: list[str] = []
        for scraped_page in scraped_pages:
            evidence = self._extract_evidence(statement, scraped_page)
            if evidence:
                evidence_pieces.append(evidence)

        # Build enriched context
        if evidence_pieces:
            enriched_context = (
                f"{statement}\n\n"
                f"Provided Evidence:\n"
                f"{chr(10).join(evidence_pieces)}"
            )
        else:
            # If no evidence was extracted, return original statement
            enriched_context = statement

        return dspy.Prediction(
            enriched_context=enriched_context,
            urls_processed=len(evidence_pieces)
        )
