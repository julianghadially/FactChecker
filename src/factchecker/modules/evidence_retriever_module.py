"""Evidence retriever module for gathering web evidence via search and scraping."""

import dspy
from src.services.serper_service import SerperService
from src.services.firecrawl_service import FirecrawlService


class EvidenceRetrieverModule(dspy.Module):
    """Module that retrieves web evidence by searching and scraping content.

    Takes search queries as input and:
    1. Searches the web using SerperService (5 results per query)
    2. Scrapes the top 3-5 results using FirecrawlService
    3. Collects markdown content from successful scrapes
    4. Returns combined evidence with source attribution

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

    def forward_from_urls(self, urls: list[str]) -> dspy.Prediction:
        """Retrieve evidence by directly scraping provided URLs (no search).

        Args:
            urls: List of URL strings to scrape directly.

        Returns:
            dspy.Prediction with:
                - evidence: Combined markdown content from all scraped URLs (with source attribution)
                - sources: List of dicts with {url, title, success} for each attempted scrape
        """
        all_evidence = []
        all_sources = []

        for url in urls:
            try:
                scraped = self.firecrawl.scrape(url, skip_pdfs=False)

                if scraped.success and scraped.markdown:
                    # Add evidence with clear source attribution
                    # Use URL as title if no title is available from scrape metadata
                    title = getattr(scraped, 'title', None) or url
                    evidence_chunk = f"## Source: {title}\nURL: {url}\n\n{scraped.markdown}\n\n---\n\n"
                    all_evidence.append(evidence_chunk)
                    all_sources.append({
                        "url": url,
                        "title": title,
                        "success": True
                    })
                else:
                    # Track failed scrapes
                    all_sources.append({
                        "url": url,
                        "title": url,
                        "success": False
                    })
            except Exception as scrape_error:
                # Handle individual scrape failures gracefully
                print(f"Failed to scrape {url}: {scrape_error}")
                all_sources.append({
                    "url": url,
                    "title": url,
                    "success": False
                })

        # Combine all evidence
        combined_evidence = "".join(all_evidence)

        # If no evidence was gathered, provide informative message
        if not combined_evidence.strip():
            combined_evidence = "No evidence could be retrieved from provided URLs."

        return dspy.Prediction(
            evidence=combined_evidence,
            sources=all_sources,
        )

    def forward(self, queries: list[str]) -> dspy.Prediction:
        """Retrieve evidence from the web for given search queries.

        Args:
            queries: List of search query strings.

        Returns:
            dspy.Prediction with:
                - evidence: Combined markdown content from all scraped pages (with source attribution)
                - sources: List of dicts with {url, title, success} for each attempted scrape
        """
        all_evidence = []
        all_sources = []

        for query in queries:
            try:
                # Search for results
                results = self.serper.search(query, num_results=5)

                # Scrape top N results
                for result in results[:self.max_results_per_query]:
                    try:
                        scraped = self.firecrawl.scrape(result.link, skip_pdfs=False)

                        if scraped.success and scraped.markdown:
                            # Add evidence with clear source attribution
                            evidence_chunk = f"## Source: {result.title}\nURL: {result.link}\n\n{scraped.markdown}\n\n---\n\n"
                            all_evidence.append(evidence_chunk)
                            all_sources.append({
                                "url": result.link,
                                "title": result.title,
                                "success": True
                            })
                        else:
                            # Track failed scrapes
                            all_sources.append({
                                "url": result.link,
                                "title": result.title,
                                "success": False
                            })
                    except Exception as scrape_error:
                        # Handle individual scrape failures gracefully
                        print(f"Failed to scrape {result.link}: {scrape_error}")
                        all_sources.append({
                            "url": result.link,
                            "title": result.title,
                            "success": False
                        })

            except Exception as search_error:
                # Handle search failures gracefully
                print(f"Failed to search for '{query}': {search_error}")
                continue

        # Combine all evidence and truncate if needed
        combined_evidence = "".join(all_evidence)

        if len(combined_evidence) > self.max_evidence_length:
            combined_evidence = combined_evidence[:self.max_evidence_length] + "\n\n[Evidence truncated due to length...]"

        # If no evidence was gathered, provide informative message
        if not combined_evidence.strip():
            combined_evidence = "No evidence could be retrieved from web sources."

        return dspy.Prediction(
            evidence=combined_evidence,
            sources=all_sources,
        )
