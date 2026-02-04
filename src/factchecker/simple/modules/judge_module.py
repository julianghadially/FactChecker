"""Simple judge module - barebones fact checker with optional web verification."""

import dspy
import re
from src.factchecker.simple.signatures.judge import Judge
from src.services.serper_service import SerperService
from src.services.firecrawl_service import FirecrawlService


class JudgeModule(dspy.Module):
    """Two-pass fact checker with optional web verification.

    First pass: Evaluates statements using LLM knowledge only.
    Second pass (triggered conditionally): Performs lightweight web research
    if the first pass is uncertain or mentions knowledge limitations.

    Trigger conditions for web verification:
    - CONTAINS_UNSUPPORTED_CLAIMS verdict with confidence < 0.6
    - Reasoning contains keywords: 'knowledge cutoff', 'cannot verify',
      'cannot confirm', '2024', '2025'

    When triggered, performs:
    1. One Serper search query (using statement or simplified version)
    2. Scrapes top 2 results with Firecrawl
    3. Re-evaluates with evidence passed to Judge

    This provides a middle ground between the barebones Judge and the full
    FactCheckerPipeline - adding web verification only when needed.
    """

    # Keywords that trigger web verification
    TRIGGER_KEYWORDS = [
        'knowledge cutoff',
        'cannot verify',
        'cannot confirm',
        '2024',
        '2025'
    ]

    def __init__(self):
        """Initialize the two-pass judge module."""
        super().__init__()
        self.judge = dspy.ChainOfThought(Judge)
        self.serper = SerperService()
        self.firecrawl = FirecrawlService()

    def _should_trigger_research(
        self,
        verdict: str,
        confidence: float,
        reasoning: str
    ) -> bool:
        """Determine if web research should be triggered.

        Args:
            verdict: The verdict from first pass.
            confidence: Confidence score from first pass.
            reasoning: Reasoning text from first pass.

        Returns:
            True if web research should be triggered, False otherwise.
        """
        # Trigger if CONTAINS_UNSUPPORTED_CLAIMS with low confidence
        if verdict == "CONTAINS_UNSUPPORTED_CLAIMS" and confidence < 0.6:
            return True

        # Trigger if reasoning mentions knowledge limitations
        reasoning_lower = reasoning.lower()
        for keyword in self.TRIGGER_KEYWORDS:
            if keyword in reasoning_lower:
                return True

        return False

    def _gather_evidence(self, statement: str) -> str:
        """Perform lightweight web research.

        Args:
            statement: The statement to research.

        Returns:
            Formatted evidence string from web sources.
        """
        try:
            # Execute one search query using the statement itself
            search_results = self.serper.search(query=statement, num_results=2)

            if not search_results:
                return ""

            # Scrape top 2 results
            evidence_parts = []
            for i, result in enumerate(search_results[:2], 1):
                scraped = self.firecrawl.scrape(
                    url=result.link,
                    max_length=5000  # Limit to keep context manageable
                )

                if scraped.success and scraped.markdown:
                    evidence_parts.append(
                        f"Source {i}: {result.title} ({result.link})\n"
                        f"{scraped.markdown}\n"
                    )

            return "\n---\n".join(evidence_parts) if evidence_parts else ""

        except Exception as e:
            # If research fails, return empty evidence
            print(f"Warning: Evidence gathering failed: {e}")
            return ""

    def forward(self, statement: str) -> dspy.Prediction:
        """Evaluate a statement for factual correctness with two-pass architecture.

        Args:
            statement: The statement to evaluate.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
                - research_triggered: Boolean indicating if second pass was used
        """
        # First pass: Judge with LLM knowledge only
        first_pass = self.judge(statement=statement, evidence="")

        # Check if we should trigger web research
        should_research = self._should_trigger_research(
            verdict=first_pass.verdict,
            confidence=first_pass.confidence,
            reasoning=first_pass.reasoning
        )

        if not should_research:
            # Return first pass result
            return dspy.Prediction(
                statement=statement,
                overall_verdict=first_pass.verdict,
                confidence=first_pass.confidence,
                reasoning=first_pass.reasoning,
                research_triggered=False
            )

        # Second pass: Gather evidence and re-evaluate
        evidence = self._gather_evidence(statement)

        if not evidence:
            # If evidence gathering failed, return first pass result
            return dspy.Prediction(
                statement=statement,
                overall_verdict=first_pass.verdict,
                confidence=first_pass.confidence,
                reasoning=first_pass.reasoning + "\n\n[Note: Web research was attempted but no evidence could be retrieved.]",
                research_triggered=True
            )

        # Judge again with evidence
        second_pass = self.judge(statement=statement, evidence=evidence)

        return dspy.Prediction(
            statement=statement,
            overall_verdict=second_pass.verdict,
            confidence=second_pass.confidence,
            reasoning=second_pass.reasoning,
            research_triggered=True
        )
