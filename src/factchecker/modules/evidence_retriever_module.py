"""Evidence retriever module for gathering web evidence via search and scraping."""

import dspy
from src.services.serper_service import SerperService
from src.services.firecrawl_service import FirecrawlService


class EvidenceRetrieverModule(dspy.Module):
    """Module that retrieves web evidence using a two-phase prioritized search strategy.

    Takes two types of queries as input and implements prioritized retrieval:
    1. PRIMARY SOURCE PHASE: Executes primary_source_queries first, scraping top 2 results
       per query from authoritative sources (e.g., official sites, index providers)
    2. GENERAL SEARCH PHASE: Then executes general_queries, scraping top 2 results per query
       for supporting context and cross-validation

    Uses SerperService for search and FirecrawlService for content extraction.
    Returns combined evidence with source attribution and tracking.

    This is the second stage of the evidence-aware fact-checking pipeline.
    """

    def __init__(self,
                 primary_results_per_query: int = 2,
                 general_results_per_query: int = 2,
                 max_evidence_length: int = 15000):
        """Initialize the evidence retriever module.

        Args:
            primary_results_per_query: Maximum number of URLs to scrape per primary source query (default 2).
            general_results_per_query: Maximum number of URLs to scrape per general query (default 2).
            max_evidence_length: Maximum total characters of evidence to return (default 15000).
        """
        super().__init__()
        self.serper = SerperService()
        self.firecrawl = FirecrawlService()
        self.primary_results_per_query = primary_results_per_query
        self.general_results_per_query = general_results_per_query
        self.max_evidence_length = max_evidence_length

    def forward(self, primary_source_queries: list[str] = None, general_queries: list[str] = None) -> dspy.Prediction:
        """Retrieve evidence using a two-phase prioritized search strategy.

        Args:
            primary_source_queries: List of site-specific queries targeting authoritative sources.
            general_queries: List of broader queries for context and verification.

        Returns:
            dspy.Prediction with:
                - evidence: Combined markdown content from all scraped pages (with source attribution)
                - sources: List of dicts with {url, title, success, query_type} for each attempted scrape
        """
        all_evidence = []
        all_sources = []

        # Handle None values
        primary_source_queries = primary_source_queries or []
        general_queries = general_queries or []

        # PHASE 1: Execute primary source queries first (prioritize authoritative evidence)
        print(f"[Phase 1] Executing {len(primary_source_queries)} primary source queries...")
        for query in primary_source_queries:
            self._process_query(
                query=query,
                query_type="primary_source",
                max_results=self.primary_results_per_query,
                all_evidence=all_evidence,
                all_sources=all_sources
            )

        # PHASE 2: Execute general queries for supporting context
        print(f"[Phase 2] Executing {len(general_queries)} general queries...")
        for query in general_queries:
            self._process_query(
                query=query,
                query_type="general",
                max_results=self.general_results_per_query,
                all_evidence=all_evidence,
                all_sources=all_sources
            )

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

    def _process_query(self, query: str, query_type: str, max_results: int,
                       all_evidence: list, all_sources: list) -> None:
        """Process a single query: search and scrape results.

        Args:
            query: The search query string.
            query_type: Type of query ("primary_source" or "general").
            max_results: Maximum number of results to scrape for this query.
            all_evidence: List to append evidence chunks to (modified in place).
            all_sources: List to append source metadata to (modified in place).
        """
        try:
            # Search for results
            results = self.serper.search(query, num_results=5)

            # Scrape top N results
            for result in results[:max_results]:
                try:
                    scraped = self.firecrawl.scrape(result.link, skip_pdfs=False)

                    if scraped.success and scraped.markdown:
                        # Add evidence with clear source attribution
                        evidence_chunk = f"## Source: {result.title}\nURL: {result.link}\nQuery Type: {query_type}\n\n{scraped.markdown}\n\n---\n\n"
                        all_evidence.append(evidence_chunk)
                        all_sources.append({
                            "url": result.link,
                            "title": result.title,
                            "success": True,
                            "query_type": query_type
                        })
                    else:
                        # Track failed scrapes
                        all_sources.append({
                            "url": result.link,
                            "title": result.title,
                            "success": False,
                            "query_type": query_type
                        })
                except Exception as scrape_error:
                    # Handle individual scrape failures gracefully
                    print(f"Failed to scrape {result.link}: {scrape_error}")
                    all_sources.append({
                        "url": result.link,
                        "title": result.title,
                        "success": False,
                        "query_type": query_type
                    })

        except Exception as search_error:
            # Handle search failures gracefully
            print(f"Failed to search for '{query}': {search_error}")
