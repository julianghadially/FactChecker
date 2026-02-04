"""Adaptive judge module with automatic fallback to full fact-checking pipeline.

This module intelligently routes fact-checking requests:
- First attempts fast judgment using JudgeModule
- If verdict is CONTAINS_UNSUPPORTED_CLAIMS with low confidence, automatically
  falls back to FactCheckerPipeline for thorough web research
- Confidence threshold acts as a natural decision boundary for when external
  verification is needed
"""

import dspy
import logging
from src.factchecker.simple.modules.judge_module import JudgeModule
from src.factchecker.modules.fact_checker_pipeline import FactCheckerPipeline

# Configure logging
logger = logging.getLogger(__name__)


class AdaptiveJudgeModule(dspy.Module):
    """Adaptive fact-checker with intelligent routing based on confidence.

    This module wraps JudgeModule with automatic fallback to FactCheckerPipeline.
    When the judge returns CONTAINS_UNSUPPORTED_CLAIMS with low confidence, it
    signals that the model is uncertain and needs external research to make an
    informed decision.

    Attributes:
        confidence_threshold: Minimum confidence to accept CONTAINS_UNSUPPORTED_CLAIMS
                            verdict without fallback (default: 0.7)
        enable_fallback: Whether to enable automatic fallback to pipeline (default: True)
        max_judge_iterations: Max iterations for FactCheckerPipeline's FireJudge
        max_page_visits: Max pages to visit per search in FactCheckerPipeline
        judge: Fast JudgeModule for initial evaluation
        pipeline: Full FactCheckerPipeline for thorough research (lazy-initialized)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        enable_fallback: bool = True,
        max_judge_iterations: int = 3,
        max_page_visits: int = 3
    ):
        """Initialize the adaptive judge module.

        Args:
            confidence_threshold: Confidence below which to trigger fallback for
                                CONTAINS_UNSUPPORTED_CLAIMS verdicts (0.0-1.0).
                                Default: 0.7
            enable_fallback: Whether to enable automatic fallback to pipeline.
                           If False, always returns JudgeModule result.
                           Default: True
            max_judge_iterations: Max search iterations per claim in pipeline.
                                Default: 3
            max_page_visits: Max pages to visit per search query in pipeline.
                           Default: 3
        """
        super().__init__()

        # Validation
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}")

        self.confidence_threshold = confidence_threshold
        self.enable_fallback = enable_fallback
        self.max_judge_iterations = max_judge_iterations
        self.max_page_visits = max_page_visits

        # Initialize fast judge module
        self.judge = JudgeModule()

        # Lazy-initialize pipeline only when needed (saves resources)
        self._pipeline = None

        logger.info(
            f"AdaptiveJudgeModule initialized with confidence_threshold={confidence_threshold}, "
            f"enable_fallback={enable_fallback}"
        )

    @property
    def pipeline(self) -> FactCheckerPipeline:
        """Lazy-initialize the full fact-checking pipeline."""
        if self._pipeline is None:
            logger.info("Initializing FactCheckerPipeline for fallback")
            self._pipeline = FactCheckerPipeline(
                max_judge_iterations=self.max_judge_iterations,
                max_page_visits=self.max_page_visits
            )
        return self._pipeline

    def forward(self, statement: str) -> dspy.Prediction:
        """Evaluate a statement with adaptive routing.

        First calls JudgeModule. If the verdict is CONTAINS_UNSUPPORTED_CLAIMS
        and confidence is below threshold, automatically falls back to
        FactCheckerPipeline for web research.

        Args:
            statement: The statement to evaluate.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
                - fallback_triggered: Boolean indicating if pipeline fallback was used
                - claims: List of claims (only present if fallback triggered)
                - claim_results: Detailed claim results (only present if fallback triggered)
        """
        logger.info(f"Evaluating statement: {statement[:100]}...")

        # Step 1: Get initial judgment from fast judge
        judge_result = self.judge(statement=statement)

        logger.info(
            f"JudgeModule verdict: {judge_result.overall_verdict}, "
            f"confidence: {judge_result.confidence:.3f}"
        )

        # Step 2: Check if fallback is needed
        needs_fallback = (
            self.enable_fallback
            and judge_result.overall_verdict == "CONTAINS_UNSUPPORTED_CLAIMS"
            and judge_result.confidence < self.confidence_threshold
        )

        if not needs_fallback:
            # Return judge result as-is
            logger.info("Returning JudgeModule result (no fallback needed)")
            return dspy.Prediction(
                statement=statement,
                overall_verdict=judge_result.overall_verdict,
                confidence=judge_result.confidence,
                reasoning=judge_result.reasoning,
                fallback_triggered=False
            )

        # Step 3: Trigger fallback to full pipeline
        logger.warning(
            f"Triggering fallback to FactCheckerPipeline: "
            f"verdict={judge_result.overall_verdict}, "
            f"confidence={judge_result.confidence:.3f} < {self.confidence_threshold}"
        )

        pipeline_result = self.pipeline(statement=statement)

        logger.info(
            f"FactCheckerPipeline verdict: {pipeline_result.overall_verdict}, "
            f"confidence: {pipeline_result.confidence:.3f}"
        )

        # Return pipeline result with fallback flag
        return dspy.Prediction(
            statement=statement,
            overall_verdict=pipeline_result.overall_verdict,
            confidence=pipeline_result.confidence,
            reasoning=pipeline_result.reasoning,
            fallback_triggered=True,
            claims=pipeline_result.claims,
            claim_results=pipeline_result.claim_results
        )
