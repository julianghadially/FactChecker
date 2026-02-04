"""Context enrichment module for fetching content from URLs."""

from typing import List, Optional
from src.services.firecrawl_service import FirecrawlService, ScrapedPage


class ContextEnrichmentModule:
    """Module for enriching context by scraping URLs using FirecrawlService.

    This module takes a list of URLs, scrapes their content using FirecrawlService,
    and combines the scraped markdown content into a single context string.
    """

    def __init__(self):
        """Initialize the context enrichment module with FirecrawlService."""
        self.firecrawl = FirecrawlService()

    def forward(self, urls: List[str]) -> str:
        """Scrape URLs and combine their content into context.

        Args:
            urls: List of URLs to scrape for context.

        Returns:
            Combined markdown content from all successfully scraped URLs.
            Returns empty string if all scraping attempts fail.
        """
        if not urls:
            return ""

        scraped_pages: List[ScrapedPage] = []
        for url in urls:
            scraped_page = self.firecrawl.scrape(url.strip())
            if scraped_page.success:
                scraped_pages.append(scraped_page)

        if not scraped_pages:
            return ""

        # Combine all scraped content into a single context string
        context_parts = []
        for page in scraped_pages:
            context_parts.append(f"## Source: {page.url}")
            if page.title:
                context_parts.append(f"### Title: {page.title}")
            context_parts.append(page.markdown)
            context_parts.append("")  # Empty line for separation

        return "\n".join(context_parts)
