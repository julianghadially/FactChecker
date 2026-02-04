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
        self.priority_urls = []  # Will be set by forward() if provided

    def forward(self, claim: str, query: str, priority_urls: list[str] = None) -> str:
        """Research a claim by searching and visiting relevant pages.

        Priority URLs (if provided) are scraped first before performing web searches,
        allowing the system to leverage provided evidence sources.

        Args:
            claim: The claim being fact-checked.
            query: Search query to execute.
            priority_urls: Optional list of URLs to scrape first before web search.

        Returns:
            Aggregated evidence from visited pages as a formatted string.
        """
        visited_urls: list[str] = []
        all_evidence: list[str] = []

        # First, process priority URLs if provided
        if priority_urls:
            print(f"Processing {len(priority_urls)} priority URLs before web search...")
            for url in priority_urls[:self.max_page_visits]:  # Limit priority URLs to max_page_visits
                if url in visited_urls:
                    continue

                visited_urls.append(url)

                # Scrape the priority URL
                scraped = self.firecrawl.scrape(url)
                if not scraped.success:
                    all_evidence.append(
                        f"[Failed to scrape priority URL {url}: {scraped.error}]"
                    )
                    continue

                # Extract relevant evidence
                summary = self.evidence_summarizer(
                    claim=claim,
                    page_content=scraped.markdown,
                    source_url=url
                )

                all_evidence.append(
                    f"Source: {url} (priority)\n"
                    f"Stance: {summary.evidence_stance}\n"
                    f"Evidence: {summary.relevant_evidence}"
                )

                # Early exit if we found strong evidence from priority URLs
                if summary.evidence_stance in ["supports", "refutes"]:
                    print(f"Found {summary.evidence_stance} evidence in priority URL, continuing to web search...")

        # If we've used all page visits on priority URLs, return what we have
        if len(visited_urls) >= self.max_page_visits:
            evidence = "\n\n".join(all_evidence) if all_evidence else "No relevant evidence found in priority URLs."
            return dspy.Prediction(evidence=evidence)

        # Execute search for additional sources
        search_results = self.serper.search(query, num_results=10)

        if not search_results:
            if all_evidence:
                # Return evidence from priority URLs
                evidence = "\n\n".join(all_evidence)
                return dspy.Prediction(evidence=evidence)
            return dspy.Prediction(evidence="No search results found.")

        # Convert to dict format for signature
        results_for_llm = [
            {"title": r.title, "link": r.link, "snippet": r.snippet}
            for r in search_results
        ]

        # Continue with remaining page visits from web search
        remaining_visits = self.max_page_visits - len(visited_urls)
        for _ in range(remaining_visits):
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

        evidence ="\n\n".join(all_evidence) if all_evidence else "No relevant evidence found."
        return dspy.Prediction(evidence=evidence)
