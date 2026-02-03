"""Simple judge module - barebones fact checker with optional web search fallback."""

import dspy
from typing import Optional
from src.factchecker.simple.signatures.judge import Judge
from src.services.serper_service import SerperService
from src.services.firecrawl_service import FirecrawlService


class JudgeModule(dspy.Module):
    """Barebones fact checker that judges statements with optional web search.

    Takes a statement as input and outputs a verdict using LLM knowledge.
    When the LLM explicitly indicates it needs external verification, it can
    automatically perform a lightweight web search to gather recent evidence.

    Two-stage architecture:
    1. First attempt judgment using parametric knowledge
    2. If the LLM sets needs_external_verification=True, trigger a focused
       web search and re-evaluate with evidence

    This serves as a simpler/faster alternative to the full FactCheckerPipeline
    for cases where minimal external research is needed.
    """

    def __init__(self, use_web_search: bool = True):
        """Initialize the simple judge module.

        Args:
            use_web_search: Whether to enable web search fallback when
                LLM detects knowledge limitations. Default is True.
        """
        super().__init__()
        self.judge = dspy.ChainOfThought(Judge)
        self.use_web_search = use_web_search

        # Initialize web services lazily only if needed
        self._serper_service: Optional[SerperService] = None
        self._firecrawl_service: Optional[FirecrawlService] = None

    @property
    def serper_service(self) -> SerperService:
        """Lazy initialization of SerperService."""
        if self._serper_service is None:
            self._serper_service = SerperService()
        return self._serper_service

    @property
    def firecrawl_service(self) -> FirecrawlService:
        """Lazy initialization of FirecrawlService."""
        if self._firecrawl_service is None:
            self._firecrawl_service = FirecrawlService()
        return self._firecrawl_service


    def _extract_search_query(self, statement: str) -> str:
        """Derive a search query from the statement.

        Args:
            statement: The statement to fact-check.

        Returns:
            A search query string optimized for verification.
        """
        # Simple heuristic: use the statement directly, optionally add "news"
        # for temporal queries. Could be enhanced with LLM-based query generation.
        return statement

    def _gather_web_evidence(self, query: str, max_results: int = 2) -> str:
        """Perform web search and scrape top results for evidence.

        Args:
            query: The search query.
            max_results: Maximum number of results to scrape (default 2).

        Returns:
            Concatenated markdown content from scraped pages.
        """
        try:
            # Perform search using SerperService
            search_results = self.serper_service.search(query, num_results=max_results)

            if not search_results:
                return "No search results found."

            # Scrape top results
            evidence_parts = []
            for i, result in enumerate(search_results[:max_results], 1):
                scraped = self.firecrawl_service.scrape(
                    result.link,
                    max_length=5000  # Limit to 5000 chars per page
                )

                if scraped.success:
                    evidence_parts.append(
                        f"### Source {i}: {result.title}\n"
                        f"URL: {result.link}\n\n"
                        f"{scraped.markdown}\n"
                    )
                else:
                    # Fall back to snippet if scraping fails
                    evidence_parts.append(
                        f"### Source {i}: {result.title}\n"
                        f"URL: {result.link}\n"
                        f"Snippet: {result.snippet}\n"
                    )

            return "\n---\n".join(evidence_parts)

        except Exception as e:
            return f"Error gathering web evidence: {str(e)}"

    def forward(self, statement: str) -> dspy.Prediction:
        """Evaluate a statement for factual correctness.

        Performs a two-stage evaluation:
        1. Initial judgment using parametric knowledge
        2. If knowledge limitations detected, perform web search and re-evaluate

        Args:
            statement: The statement to evaluate.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
                - web_evidence_used: Boolean indicating if web search was performed
        """
        # Stage 1: Initial judgment with parametric knowledge
        result = self.judge(statement=statement)
        web_evidence_used = False

        # Stage 2: Check if web search is needed and enabled
        if self.use_web_search and result.needs_external_verification:
            # Derive search query from statement
            query = self._extract_search_query(statement)

            # Gather web evidence
            web_evidence = self._gather_web_evidence(query)

            # Re-evaluate with evidence appended to context
            statement_with_evidence = (
                f"{statement}\n\n"
                f"--- Web Evidence ---\n"
                f"{web_evidence}"
            )

            # Re-run judgment with evidence
            result = self.judge(statement=statement_with_evidence)
            web_evidence_used = True

        return dspy.Prediction(
            statement=statement,
            overall_verdict=result.verdict,
            confidence=result.confidence,
            reasoning=result.reasoning,
            web_evidence_used=web_evidence_used,
        )
