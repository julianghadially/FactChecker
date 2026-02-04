"""Simple judge module - barebones fact checker without research."""

import dspy
import re
from typing import TYPE_CHECKING
from src.factchecker.simple.signatures.judge import Judge

if TYPE_CHECKING:
    from src.factchecker.modules.fact_checker_pipeline import FactCheckerPipeline


class JudgeModule(dspy.Module):
    """Intelligent hybrid fact checker that routes to appropriate strategy.

    This module automatically detects when external evidence is needed and delegates
    appropriately:
    - If URLs are provided or temporal claims (2024+), uses FactCheckerPipeline for web research
    - Otherwise, uses fast internal ChainOfThought judge based on LLM knowledge

    This serves as a smart dispatcher that chooses the optimal fact-checking approach
    based on input characteristics.
    """

    def __init__(self, enable_pipeline: bool = True):
        """Initialize the intelligent hybrid judge module.

        Args:
            enable_pipeline: Whether to enable FactCheckerPipeline delegation.
                           If False, always uses internal judge (for testing/performance).
        """
        super().__init__()
        self.judge = dspy.ChainOfThought(Judge)
        self._pipeline = None
        self._enable_pipeline = enable_pipeline

    @property
    def pipeline(self) -> "FactCheckerPipeline":
        """Lazy initialization of FactCheckerPipeline to avoid overhead when not needed."""
        if self._pipeline is None:
            from src.factchecker.modules.fact_checker_pipeline import FactCheckerPipeline
            self._pipeline = FactCheckerPipeline()
        return self._pipeline

    def _should_use_pipeline(
        self,
        urls: list[str] | None,
        date_generated: str | None
    ) -> bool:
        """Determine if external research pipeline should be used.

        Args:
            urls: Optional list of URLs provided with the statement.
            date_generated: Optional date string indicating when the claim was made.

        Returns:
            True if FactCheckerPipeline should be used, False for internal judge.
        """
        if not self._enable_pipeline:
            return False

        # If URLs provided, use pipeline for external verification
        if urls and len(urls) > 0:
            return True

        # If date indicates recent claim (2024+), use pipeline
        if date_generated:
            # Extract year from various date formats
            year_match = re.search(r'202[4-9]|20[3-9]\d', date_generated)
            if year_match:
                year = int(year_match.group())
                if year >= 2024:
                    return True

        return False

    def forward(
        self,
        statement: str,
        urls: list[str] | None = None,
        date_generated: str | None = None
    ) -> dspy.Prediction:
        """Evaluate a statement for factual correctness with intelligent routing.

        This method automatically detects when external evidence is needed:
        - If URLs are provided, delegates to FactCheckerPipeline for web research
        - If date_generated indicates recent claim (2024+), uses FactCheckerPipeline
        - Otherwise, uses fast internal ChainOfThought judge

        Args:
            statement: The statement to evaluate.
            urls: Optional list of URLs related to the statement.
            date_generated: Optional date string indicating when the claim was made.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
        """
        # Route to appropriate fact-checking strategy
        if self._should_use_pipeline(urls, date_generated):
            # Use full pipeline with web research
            result = self.pipeline.forward(statement=statement)

            return dspy.Prediction(
                statement=statement,
                overall_verdict=result.overall_verdict,
                confidence=result.confidence,
                reasoning=result.reasoning,
            )
        else:
            # Use fast internal judge
            result = self.judge(statement=statement)

            return dspy.Prediction(
                statement=statement,
                overall_verdict=result.verdict,
                confidence=result.confidence,
                reasoning=result.reasoning,
            )
