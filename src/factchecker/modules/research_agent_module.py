"""Research agent module for web-based evidence gathering."""

import dspy
from ..signatures.page_selector import PageSelector
from ..signatures.evidence_summarizer import EvidenceSummarizer
from services.serper_service import SerperService
from services.firecrawl_service import FirecrawlService


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
        serper_service: SerperService,
        firecrawl_service: FirecrawlService,
        max_page_visits: int = 3
    ):
        """Initialize the research agent module.

        Args:
            serper_service: Service for executing web searches.
            firecrawl_service: Service for scraping web pages.
            max_page_visits: Maximum pages to visit per query (default 3).
        """
        super().__init__()
        self.serper = serper_service
        self.firecrawl = firecrawl_service
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
        # Execute search
        search_results = self.serper.search(query, num_results=10)

        if not search_results:
            return "No search results found."

        # Convert to dict format for signature
        results_for_llm = [
            {"title": r.title, "link": r.link, "snippet": r.snippet}
            for r in search_results
        ]

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

            all_evidence.append(
                f"Source: {selection.selected_url}\n"
                f"Stance: {summary.evidence_stance}\n"
                f"Evidence: {summary.relevant_evidence}"
            )

            # Early exit if we found strong supporting/refuting evidence
            if summary.evidence_stance in ["supports", "refutes"]:
                break

        return "\n\n".join(all_evidence) if all_evidence else "No relevant evidence found."
