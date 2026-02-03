"""Research agent module for web-based evidence gathering."""

import dspy
from src.factchecker.signatures.page_selector import PageSelector
from src.factchecker.signatures.evidence_summarizer import EvidenceSummarizer
from src.services.serper_service import SerperService
from src.services.firecrawl_service import FirecrawlService


class ResearchAgentModule(dspy.Module):
    """Research agent that searches the web and extracts relevant evidence.

    Uses LLM to intelligently select which pages to visit from search results,
    limited to a configurable number of page visits per query for cost efficiency.

    Attributes:
        max_page_visits: Maximum number of pages to visit per search query.
        serper: Service for web search.
        firecrawl: Service for page scraping.
    """

    def __init__(
        self,
        max_page_visits: int = 3
    ):
        """Initialize the research agent module.

        Args:
            serper_service: Service for executing web searches.
            firecrawl_service: Service for scraping web pages.
            max_page_visits: Maximum pages to visit per query (default 3).
        """
        super().__init__()
        self.serper = SerperService()
        self.firecrawl = FirecrawlService()
        self.max_page_visits = max_page_visits
        self.page_selector = dspy.ChainOfThought(PageSelector)
        self.evidence_summarizer = dspy.ChainOfThought(EvidenceSummarizer)

    def forward(self, claim: str, query: str) -> str:
        """Research a claim by searching and visiting relevant pages.

        Args:
            claim: The claim being fact-checked.
            query: Search query to execute.

        Returns:
            Aggregated evidence from visited pages as a formatted string.
        """
        # Execute parallel searches: regular and news (with recency filter)
        search_results = self.serper.search(query, num_results=10)
        news_results = self.serper.search_news(query, recency="m")

        # Convert regular search results to dict format
        results_for_llm = [
            {"title": r.title, "link": r.link, "snippet": r.snippet}
            for r in search_results
        ]

        # Merge news results into the results list
        # News results already come as dicts with title, link, snippet, date, source
        for news in news_results:
            # Add temporal indicator to snippet
            date_info = news.get("date", "")
            source_info = news.get("source", "")
            snippet = news.get("snippet", "")

            # Enrich snippet with temporal metadata
            enriched_snippet = f"[Recent: {date_info}] {snippet}" if date_info else snippet
            if source_info:
                enriched_snippet = f"{enriched_snippet} (Source: {source_info})"

            results_for_llm.append({
                "title": news.get("title", ""),
                "link": news.get("link", ""),
                "snippet": enriched_snippet
            })

        if not results_for_llm:
            return "No search results found."

        visited_urls: list[str] = []
        all_evidence: list[str] = []

        for _ in range(self.max_page_visits):
            # LLM selects next page to visit
            selection = self.page_selector(
                claim=claim,
                search_results=results_for_llm,
                visited_urls=visited_urls,
                current_evidence="\n".join(all_evidence)
            )

            if not selection.selected_url:
                break  # No more useful pages

            visited_urls.append(selection.selected_url)

            # Scrape the selected page
            scraped = self.firecrawl.scrape(selection.selected_url)
            if not scraped.success:
                all_evidence.append(
                    f"[Failed to scrape {selection.selected_url}: {scraped.error}]"
                )
                continue

            # Extract relevant evidence
            summary = self.evidence_summarizer(
                claim=claim,
                page_content=scraped.markdown,
                source_url=selection.selected_url
            )

            # Include temporal metadata in the evidence string
            evidence_entry = (
                f"Source: {selection.selected_url}\n"
                f"Stance: {summary.evidence_stance}\n"
                f"Evidence: {summary.relevant_evidence}\n"
                f"Temporal Context: {summary.temporal_context}"
            )
            all_evidence.append(evidence_entry)

            # Early exit if we found strong supporting/refuting evidence
            if summary.evidence_stance in ["supports", "refutes"]:
                break

        evidence ="\n\n".join(all_evidence) if all_evidence else "No relevant evidence found."
        return dspy.Prediction(evidence=evidence)
