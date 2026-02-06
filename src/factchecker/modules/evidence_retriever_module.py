"""Evidence retriever module for gathering web evidence via search and scraping."""

import dspy
from src.services.serper_service import SerperService
from src.services.firecrawl_service import FirecrawlService


class EvidenceRetrieverModule(dspy.Module):
    """Module that retrieves web evidence by searching and scraping content.

    Implements a two-stage iterative evidence gathering strategy:

    Stage 1 (Initial Search):
    1. Searches the web using SerperService (5 results per query)
    2. Scrapes the top 5 results using FirecrawlService
    3. Collects markdown content from successful scrapes

    Stage 2 (Iterative Deep Dive - if enabled and evidence insufficient):
    1. Analyzes evidence quality (successful scrapes and content length)
    2. If insufficient (<5 scrapes OR <5000 chars), generates specialized queries
    3. Targets authoritative sources (government, industry reports, official docs)
    4. Scrapes 2-3 additional results per specialized query

    Returns combined evidence with source attribution.

    This is the second stage of the evidence-aware fact-checking pipeline.
    """

    def __init__(self, max_results_per_query: int = 5, max_evidence_length: int = 15000, enable_iterative_search: bool = True):
        """Initialize the evidence retriever module.

        Args:
            max_results_per_query: Maximum number of URLs to scrape per query (default 5).
            max_evidence_length: Maximum total characters of evidence to return (default 15000).
            enable_iterative_search: Enable iterative deep dive search when initial evidence is insufficient (default True).
        """
        super().__init__()
        self.serper = SerperService()
        self.firecrawl = FirecrawlService()
        self.max_results_per_query = max_results_per_query
        self.max_evidence_length = max_evidence_length
        self.enable_iterative_search = enable_iterative_search

    def forward(self, queries: list[str]) -> dspy.Prediction:
        """Retrieve evidence from the web for given search queries.

        Implements a two-stage retrieval process:
        - Stage 1: Initial search across all queries
        - Stage 2: Iterative deep dive with specialized queries if evidence is insufficient

        Args:
            queries: List of search query strings.

        Returns:
            dspy.Prediction with:
                - evidence: Combined markdown content from all scraped pages (with source attribution)
                - sources: List of dicts with {url, title, success} for each attempted scrape
        """
        all_evidence = []
        all_sources = []

        # STAGE 1: Initial Search - Execute all queries and scrape top results
        print("Stage 1: Initial evidence gathering...")
        for query in queries:
            try:
                # Search for results
                results = self.serper.search(query, num_results=5)

                # Scrape top N results
                for result in results[:self.max_results_per_query]:
                    try:
                        scraped = self.firecrawl.scrape(result.link)

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

        # STAGE 2: Iterative Deep Dive - Check if evidence is insufficient and gather more
        if self.enable_iterative_search:
            successful_scrapes = sum(1 for s in all_sources if s["success"])
            combined_evidence_length = sum(len(e) for e in all_evidence)

            evidence_insufficient = (
                successful_scrapes < 5 or
                combined_evidence_length < 5000
            )

            if evidence_insufficient:
                print(f"Stage 2: Evidence insufficient (scrapes: {successful_scrapes}, chars: {combined_evidence_length})")
                print("Initiating deep dive with authoritative sources...")

                # Generate deep dive queries targeting authoritative sources
                deep_dive_queries = self._generate_deep_dive_queries(queries)

                for deep_query in deep_dive_queries:
                    try:
                        # Search with specialized query
                        results = self.serper.search(deep_query, num_results=5)

                        # Scrape 2-3 results from specialized queries
                        scrape_limit = min(3, len(results))
                        for result in results[:scrape_limit]:
                            try:
                                scraped = self.firecrawl.scrape(result.link)

                                if scraped.success and scraped.markdown:
                                    evidence_chunk = f"## Source: {result.title}\nURL: {result.link}\n\n{scraped.markdown}\n\n---\n\n"
                                    all_evidence.append(evidence_chunk)
                                    all_sources.append({
                                        "url": result.link,
                                        "title": result.title,
                                        "success": True
                                    })
                                else:
                                    all_sources.append({
                                        "url": result.link,
                                        "title": result.title,
                                        "success": False
                                    })
                            except Exception as scrape_error:
                                print(f"Failed to scrape {result.link}: {scrape_error}")
                                all_sources.append({
                                    "url": result.link,
                                    "title": result.title,
                                    "success": False
                                })

                    except Exception as search_error:
                        print(f"Failed deep dive search for '{deep_query}': {search_error}")
                        continue

                final_scrapes = sum(1 for s in all_sources if s["success"])
                final_length = sum(len(e) for e in all_evidence)
                print(f"Stage 2 complete: Total scrapes: {final_scrapes}, Total chars: {final_length}")

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

    def _generate_deep_dive_queries(self, original_queries: list[str]) -> list[str]:
        """Generate specialized deep dive queries targeting authoritative sources.

        Args:
            original_queries: The original search queries from Stage 1.

        Returns:
            List of 2 specialized queries with site-specific or industry-specific terms.
        """
        if not original_queries:
            return []

        deep_dive_queries = []

        # Take the first query as base (most relevant to the claim)
        base_query = original_queries[0]

        # Strategy 1: Government/Official sources with site restrictions
        gov_query = f"{base_query} site:.gov OR site:.edu"
        deep_dive_queries.append(gov_query)

        # Strategy 2: Industry-specific authoritative terms
        # Add terms that signal official documentation, reports, inspections
        authoritative_terms = [
            "official report",
            "inspection data",
            "regulatory filing",
            "government database",
            "industry specification",
            "capacity certification",
            "technical documentation"
        ]

        # Select terms based on query content
        selected_term = "official report"  # Default
        if "capacity" in base_query.lower() or "production" in base_query.lower():
            selected_term = "capacity certification"
        elif "inspection" in base_query.lower() or "safety" in base_query.lower():
            selected_term = "inspection data"
        elif "company" in base_query.lower() or "facility" in base_query.lower():
            selected_term = "regulatory filing"

        industry_query = f"{base_query} {selected_term}"
        deep_dive_queries.append(industry_query)

        return deep_dive_queries
