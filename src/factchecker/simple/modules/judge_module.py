"""Simple judge module - barebones fact checker without research."""

import dspy
import re
from typing import Optional
from src.factchecker.simple.signatures.judge import Judge
from src.services.serper_service import SerperService
from src.services.firecrawl_service import FirecrawlService


class JudgeModule(dspy.Module):
    """Adaptive fact checker that judges statements with optional web search.

    Takes a statement as input and outputs a verdict using LLM knowledge.
    If the initial judgment indicates uncertainty (mentions "knowledge cutoff",
    "cannot verify", "cannot confirm", or has confidence < 0.6), the module
    automatically performs web search to gather evidence and makes a final
    judgment with the additional context.

    This serves as a simpler/faster alternative to the full FactCheckerPipeline
    for cases where external research is not always needed, but provides
    adaptive search for recent events or uncertain claims.
    """

    def __init__(
        self,
        enable_adaptive_search: bool = True,
        confidence_threshold: float = 0.6,
        num_search_results: int = 3,
        max_scrape_length: int = 8000
    ):
        """Initialize the adaptive judge module.

        Args:
            enable_adaptive_search: Whether to enable automatic web search on uncertainty.
            confidence_threshold: Confidence below this triggers web search (default: 0.6).
            num_search_results: Number of search results to scrape (default: 3).
            max_scrape_length: Maximum characters per scraped page (default: 8000).
        """
        super().__init__()
        self.judge = dspy.ChainOfThought(Judge)
        self.enable_adaptive_search = enable_adaptive_search
        self.confidence_threshold = confidence_threshold
        self.num_search_results = num_search_results
        self.max_scrape_length = max_scrape_length

        # Initialize services lazily to avoid errors if API keys are not set
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

    def _should_trigger_search(self, reasoning: str, confidence: float) -> bool:
        """Determine if web search should be triggered based on initial judgment.

        Args:
            reasoning: The reasoning text from the initial judgment.
            confidence: The confidence score from the initial judgment.

        Returns:
            True if web search should be triggered, False otherwise.
        """
        if not self.enable_adaptive_search:
            return False

        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return True

        # Check for uncertainty phrases in reasoning (case-insensitive)
        uncertainty_patterns = [
            r"knowledge cutoff",
            r"cannot verify",
            r"cannot confirm",
            r"unable to verify",
            r"unable to confirm",
            r"no access to.*information",
            r"beyond my knowledge",
            r"don't have.*information",
            r"insufficient information"
        ]

        reasoning_lower = reasoning.lower()
        for pattern in uncertainty_patterns:
            if re.search(pattern, reasoning_lower):
                return True

        return False

    def _gather_web_evidence(self, statement: str) -> str:
        """Gather web evidence by searching and scraping top results.

        Args:
            statement: The statement to search for.

        Returns:
            Formatted evidence string from scraped web pages.
        """
        try:
            # Search for the statement/claim
            search_results = self.serper_service.search(
                query=statement,
                num_results=self.num_search_results
            )

            if not search_results:
                return "No search results found."

            evidence_parts = []
            # Scrape top results
            for i, result in enumerate(search_results[:self.num_search_results], 1):
                scraped = self.firecrawl_service.scrape(
                    url=result.link,
                    max_length=self.max_scrape_length
                )

                if scraped.success:
                    evidence_parts.append(
                        f"=== Source {i}: {result.title} ===\n"
                        f"URL: {result.link}\n"
                        f"Content:\n{scraped.markdown}\n"
                    )
                else:
                    evidence_parts.append(
                        f"=== Source {i}: {result.title} ===\n"
                        f"URL: {result.link}\n"
                        f"Snippet: {result.snippet}\n"
                        f"(Full content unavailable: {scraped.error})\n"
                    )

            return "\n".join(evidence_parts)

        except Exception as e:
            return f"Error gathering web evidence: {str(e)}"

    def forward(self, statement: str) -> dspy.Prediction:
        """Evaluate a statement for factual correctness with adaptive web search.

        Args:
            statement: The statement to evaluate.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
                - web_search_triggered: Boolean indicating if web search was used
                - evidence: The web evidence gathered (if any)
        """
        # Initial judgment without evidence
        initial_result = self.judge(statement=statement, evidence="")

        # Check if we should trigger web search
        should_search = self._should_trigger_search(
            initial_result.reasoning,
            initial_result.confidence
        )

        if should_search:
            print(f"[JudgeModule] Initial confidence ({initial_result.confidence:.2f}) below threshold "
                  f"or uncertainty detected. Triggering web search...")

            # Gather web evidence
            evidence = self._gather_web_evidence(statement)

            # Make final judgment with evidence
            final_result = self.judge(statement=statement, evidence=evidence)

            return dspy.Prediction(
                statement=statement,
                overall_verdict=final_result.verdict,
                confidence=final_result.confidence,
                reasoning=final_result.reasoning,
                web_search_triggered=True,
                evidence=evidence,
                initial_confidence=initial_result.confidence,
                initial_reasoning=initial_result.reasoning
            )
        else:
            # Return initial judgment without web search
            return dspy.Prediction(
                statement=statement,
                overall_verdict=initial_result.verdict,
                confidence=initial_result.confidence,
                reasoning=initial_result.reasoning,
                web_search_triggered=False,
                evidence=""
            )
