"""Simple judge module - barebones fact checker with optional web search fallback."""

import dspy
from src.factchecker.simple.signatures.judge import Judge
from src.factchecker.simple.signatures.web_augmented_judge import WebAugmentedJudge
from src.services.serper_service import SerperService
from src.services.firecrawl_service import FirecrawlService
import re


class JudgeModule(dspy.Module):
    """Hybrid fact checker with LLM-first approach and web search fallback.

    Two-stage pipeline:
    1. LLM-only judgment: Attempts to evaluate using model's knowledge
    2. Web-augmented judgment: Falls back to web search if confidence < 0.7
       OR if reasoning mentions knowledge cutoff/lacking information

    This solves the "knowledge cutoff" problem while maintaining speed for
    statements the LLM can confidently verify without external research.
    """

    def __init__(self):
        """Initialize the judge module with optional web search capability."""
        super().__init__()
        self.judge = dspy.ChainOfThought(Judge)
        self.web_judge = dspy.ChainOfThought(WebAugmentedJudge)
        self.serper = SerperService()
        self.firecrawl = FirecrawlService()

    def forward(self, statement: str, web_search_enabled: bool = True) -> dspy.Prediction:
        """Evaluate a statement for factual correctness with optional web search fallback.

        Two-stage pipeline:
        1. Stage 1 (LLM-only): Attempt evaluation using model knowledge
        2. Stage 2 (Web-augmented): If uncertain, search web and re-evaluate

        Fallback triggers when:
        - Confidence score < 0.7, OR
        - Reasoning mentions: knowledge cutoff, lacking information, unable to verify, etc.

        Args:
            statement: The statement to evaluate.
            web_search_enabled: Enable web search fallback for uncertain judgments (default: True).

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
                - used_web_search: Boolean indicating if web search was used
                - evidence: Retrieved evidence (only if web search was used)
        """
        # Stage 1: LLM-only judgment
        result = self.judge(statement=statement)

        # Check if we need to fall back to web search
        needs_web_search = False

        if web_search_enabled:
            # Check confidence threshold
            if result.confidence < 0.7:
                needs_web_search = True

            # Check for uncertainty indicators in reasoning
            uncertainty_patterns = [
                r"knowledge cutoff",
                r"cutoff date",
                r"lack(?:ing)?\s+(?:sufficient\s+)?information",
                r"unable to verify",
                r"cannot (?:confirm|verify)",
                r"don't have (?:access|information)",
                r"no (?:current|recent|up-to-date) information",
                r"(?:as of|beyond) my (?:knowledge|training)",
                r"need(?:s)? more (?:recent|current|up-to-date) (?:information|data)",
            ]

            reasoning_lower = result.reasoning.lower()
            for pattern in uncertainty_patterns:
                if re.search(pattern, reasoning_lower, re.IGNORECASE):
                    needs_web_search = True
                    break

        # Stage 2: Web-augmented judgment if needed
        if needs_web_search:
            evidence = self._gather_web_evidence(statement)

            if evidence:
                # Re-evaluate with web evidence
                web_result = self.web_judge(statement=statement, evidence=evidence)

                return dspy.Prediction(
                    statement=statement,
                    overall_verdict=web_result.verdict,
                    confidence=web_result.confidence,
                    reasoning=web_result.reasoning,
                    used_web_search=True,
                    evidence=evidence,
                )

        # Return LLM-only result if web search not needed or failed
        return dspy.Prediction(
            statement=statement,
            overall_verdict=result.verdict,
            confidence=result.confidence,
            reasoning=result.reasoning,
            used_web_search=False,
            evidence=None,
        )

    def _gather_web_evidence(self, statement: str, num_results: int = 3) -> str:
        """Gather evidence from web search and scraping.

        Args:
            statement: The statement to search for evidence about.
            num_results: Number of top search results to scrape (default: 3).

        Returns:
            Formatted evidence string combining search results and scraped content.
            Returns empty string if search/scraping fails.
        """
        try:
            # Search for relevant pages
            print(f"[JudgeModule] Triggering web search for: {statement}")
            search_results = self.serper.search(query=statement, num_results=num_results)

            if not search_results:
                return ""

            # Scrape top results
            evidence_parts = []
            evidence_parts.append("=== WEB SEARCH RESULTS ===\n")

            for i, result in enumerate(search_results[:num_results], 1):
                evidence_parts.append(f"\n--- Source {i}: {result.title} ---")
                evidence_parts.append(f"URL: {result.link}")
                evidence_parts.append(f"Snippet: {result.snippet}\n")

                # Scrape the page for full content
                scraped = self.firecrawl.scrape(url=result.link, max_length=3000)

                if scraped.success and scraped.markdown:
                    evidence_parts.append("Full Content (truncated):")
                    evidence_parts.append(scraped.markdown)
                else:
                    evidence_parts.append(f"[Could not scrape content: {scraped.error}]")

            return "\n".join(evidence_parts)

        except Exception as e:
            print(f"[JudgeModule] Web evidence gathering failed: {e}")
            return ""
