"""Firecrawl API service for web page scraping."""

from dataclasses import dataclass
from typing import Optional
from firecrawl import Firecrawl
from src.context_.context import firecrawl_key
from src.tools.general_tools import clean_llm_outputted_url

@dataclass
class ScrapedPage:
    """Result from scraping a web page."""
    url: str
    markdown: str
    title: Optional[str]
    success: bool
    error: Optional[str] = None


class FirecrawlService:
    """Service for web page scraping via Firecrawl API.

    Attributes:
        client: Firecrawl client instance.
    """

    def __init__(self):
        """Initialize the Firecrawl service.

        Args:
            api_key: Firecrawl API key.
        """
        self.client = Firecrawl(api_key=firecrawl_key)

    def scrape(
        self,
        url: str,
        max_length: int = 10000
    ) -> ScrapedPage:
        """Scrape a URL and return markdown content.

        Args:
            url: URL to scrape.
            max_length: Maximum characters to return (truncate if longer).

        Returns:
            ScrapedPage with markdown content or error information.
        """
        try:
            url = clean_llm_outputted_url(url)
            result = self.client.scrape(url, formats=["markdown"])
            #result = client.scrape(url, formats=["markdown"])
            markdown = result.markdown
            # Truncate if needed to manage token costs
            if len(markdown) > max_length:
                markdown = markdown[:max_length] + "\n\n[Content truncated...]"

            return ScrapedPage(
                url=url,
                markdown=markdown,
                title=result.metadata.title,
                success=True
            )
        except Exception as e:
            return ScrapedPage(
                url=url,
                markdown="",
                title=None,
                success=False,
                error=str(e)
            )
