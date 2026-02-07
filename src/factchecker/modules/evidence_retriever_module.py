"""Evidence retriever module for gathering web evidence via search and scraping."""

import dspy
from src.services.serper_service import SerperService
from src.services.firecrawl_service import FirecrawlService


class EvidenceRetrieverModule(dspy.Module):
    """Module that retrieves web evidence by searching and scraping content.

    Takes search queries as input and:
    1. Searches the web using SerperService (5 results per query)
    2. Scrapes the top 3-5 results using FirecrawlService
    3. Returns structured source data (url, title, markdown, success) for each source

    The structured output allows downstream modules (like SourcePrioritizationModule)
    to intelligently rank and concatenate sources based on relevance.

    This is the second stage of the evidence-aware fact-checking pipeline.
    """

    def __init__(self, max_results_per_query: int = 3, max_evidence_length: int = 15000):
        """Initialize the evidence retriever module.

        Args:
            max_results_per_query: Maximum number of URLs to scrape per query (default 3).
            max_evidence_length: Maximum total characters of evidence to return (default 15000).
        """
        super().__init__()
        self.serper = SerperService()
        self.firecrawl = FirecrawlService()
        self.max_results_per_query = max_results_per_query
        self.max_evidence_length = max_evidence_length

    def forward(self, queries: list[str]) -> dspy.Prediction:
        """Retrieve evidence from the web for given search queries.

        Args:
            queries: List of search query strings.

        Returns:
            dspy.Prediction with:
                - sources: List of dicts with {url, title, markdown, success} for each attempted scrape
        """
        all_sources = []

        for query in queries:
            try:
                # Search for results
                results = self.serper.search(query, num_results=5)

                # Scrape top N results
                for result in results[:self.max_results_per_query]:
                    try:
                        scraped = self.firecrawl.scrape(result.link)

                        if scraped.success and scraped.markdown:
                            # Store structured source data with markdown content
                            all_sources.append({
                                "url": result.link,
                                "title": result.title,
                                "markdown": scraped.markdown,
                                "success": True
                            })
                        else:
                            # Track failed scrapes
                            all_sources.append({
                                "url": result.link,
                                "title": result.title,
                                "markdown": "",
                                "success": False
                            })
                    except Exception as scrape_error:
                        # Handle individual scrape failures gracefully
                        print(f"Failed to scrape {result.link}: {scrape_error}")
                        all_sources.append({
                            "url": result.link,
                            "title": result.title,
                            "markdown": "",
                            "success": False
                        })

            except Exception as search_error:
                # Handle search failures gracefully
                print(f"Failed to search for '{query}': {search_error}")
                continue

        return dspy.Prediction(
            sources=all_sources,
        )
