"""Smart routing module that intelligently delegates between JudgeModule and FactCheckerPipeline."""

import dspy
from typing import Optional
from src.factchecker.simple.modules.judge_module import JudgeModule
from src.factchecker.modules.fact_checker_pipeline import FactCheckerPipeline
from src.factchecker.signatures.temporal_detector import TemporalDetector
from src.services.firecrawl_service import FirecrawlService


class SmartJudgeModule(dspy.Module):
    """Intelligent routing module for fact-checking with automatic delegation.

    This module serves as the primary entry point for fact-checking and automatically
    routes statements to the appropriate fact-checking strategy:

    1. URL-based routing: When URLs are provided, pre-seeds FactCheckerPipeline
    2. Temporal detection: Routes recent/future claims to web research
    3. Confidence-based fallback: Falls back to web research for low-confidence verdicts
    4. Simple judge: Uses fast JudgeModule for high-confidence, non-temporal claims

    Attributes:
        judge_module: Simple judge for direct LLM evaluation
        pipeline: Full fact-checking pipeline with web research
        temporal_detector: Detector for time-sensitive claims
        firecrawl: Service for scraping provided URLs
        confidence_threshold: Minimum confidence to avoid fallback (default 0.6)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        max_judge_iterations: int = 3,
        max_page_visits: int = 3
    ):
        """Initialize the smart judge module.

        Args:
            confidence_threshold: Minimum confidence to trust JudgeModule verdict (default 0.6)
            max_judge_iterations: Max iterations for FactCheckerPipeline (default 3)
            max_page_visits: Max pages to visit per search query (default 3)
        """
        super().__init__()
        self.judge_module = JudgeModule()
        self.pipeline = FactCheckerPipeline(
            max_judge_iterations=max_judge_iterations,
            max_page_visits=max_page_visits
        )
        self.temporal_detector = dspy.ChainOfThought(TemporalDetector)
        self.firecrawl = FirecrawlService()
        self.confidence_threshold = confidence_threshold

    def _scrape_urls_as_evidence(self, urls: list[str]) -> str:
        """Scrape provided URLs and format them as initial evidence.

        Args:
            urls: List of URLs to scrape for evidence

        Returns:
            Formatted evidence string from scraped URLs
        """
        evidence_parts = []

        for url in urls:
            scraped = self.firecrawl.scrape(url)
            if scraped.success:
                evidence_parts.append(
                    f"--- Pre-seeded Evidence from {url} ---\n"
                    f"Title: {scraped.title or 'N/A'}\n"
                    f"Content: {scraped.markdown}"
                )
            else:
                evidence_parts.append(
                    f"--- Failed to scrape {url} ---\n"
                    f"Error: {scraped.error}"
                )

        return "\n\n".join(evidence_parts) if evidence_parts else ""

    def _detect_temporal_claim(self, statement: str) -> bool:
        """Detect if statement contains temporal claims requiring recent knowledge.

        Args:
            statement: The statement to analyze

        Returns:
            True if the statement requires knowledge after June 2024 or refers to current/future events
        """
        result = self.temporal_detector(statement=statement)
        return result.requires_recent_knowledge

    def forward(self, statement: str, urls: Optional[list[str]] = None) -> dspy.Prediction:
        """Execute intelligent fact-checking with automatic routing.

        Args:
            statement: The statement to fact-check
            urls: Optional list of URLs to use as evidence sources

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
                - routing_decision: String describing which path was taken
        """
        routing_decision = ""

        # Route 1: URLs provided - pre-seed pipeline with scraped evidence
        if urls:
            routing_decision = f"URLs provided ({len(urls)} URLs) - routing to FactCheckerPipeline with pre-seeded evidence"
            print(f"[SmartJudgeModule] {routing_decision}")

            initial_evidence = self._scrape_urls_as_evidence(urls)
            result = self.pipeline(statement=statement, initial_evidence=initial_evidence)

            return dspy.Prediction(
                statement=statement,
                overall_verdict=result.overall_verdict,
                confidence=result.confidence,
                reasoning=result.reasoning,
                routing_decision=routing_decision,
                claims=result.claims,
                claim_results=result.claim_results
            )

        # Route 2: Check for temporal claims requiring recent knowledge
        is_temporal = self._detect_temporal_claim(statement)
        if is_temporal:
            routing_decision = "Temporal claim detected (recent/future dates) - routing to FactCheckerPipeline for web research"
            print(f"[SmartJudgeModule] {routing_decision}")

            result = self.pipeline(statement=statement)

            return dspy.Prediction(
                statement=statement,
                overall_verdict=result.overall_verdict,
                confidence=result.confidence,
                reasoning=result.reasoning,
                routing_decision=routing_decision,
                claims=result.claims,
                claim_results=result.claim_results
            )

        # Route 3: Try simple judge first, with confidence-based fallback
        routing_decision = "No URLs or temporal claims - trying JudgeModule first"
        print(f"[SmartJudgeModule] {routing_decision}")

        judge_result = self.judge_module(statement=statement)

        # Check if we need to fall back to pipeline
        needs_fallback = (
            judge_result.confidence < self.confidence_threshold or
            judge_result.overall_verdict == "CONTAINS_UNSUPPORTED_CLAIMS"
        )

        if needs_fallback:
            fallback_reason = (
                f"low confidence ({judge_result.confidence:.2f} < {self.confidence_threshold})"
                if judge_result.confidence < self.confidence_threshold
                else f"verdict is {judge_result.overall_verdict}"
            )
            routing_decision += f" -> Falling back to FactCheckerPipeline ({fallback_reason})"
            print(f"[SmartJudgeModule] {routing_decision}")

            result = self.pipeline(statement=statement)

            return dspy.Prediction(
                statement=statement,
                overall_verdict=result.overall_verdict,
                confidence=result.confidence,
                reasoning=result.reasoning,
                routing_decision=routing_decision,
                claims=result.claims,
                claim_results=result.claim_results
            )

        # High confidence from simple judge - use that result
        routing_decision += f" -> High confidence ({judge_result.confidence:.2f}) - using JudgeModule result"
        print(f"[SmartJudgeModule] {routing_decision}")

        return dspy.Prediction(
            statement=statement,
            overall_verdict=judge_result.overall_verdict,
            confidence=judge_result.confidence,
            reasoning=judge_result.reasoning,
            routing_decision=routing_decision
        )
