"""Hybrid judge module that intelligently routes claims to web search or simple judge."""

import dspy
from src.factchecker.simple.modules.judge_module import JudgeModule
from src.factchecker.modules.fact_checker_pipeline import FactCheckerPipeline
from src.factchecker.signatures.temporal_detector import TemporalDetector


class HybridJudgeModule(dspy.Module):
    """Hybrid fact-checking module with intelligent routing.

    This module analyzes each statement to determine if it requires web-based
    verification (for temporal claims, recent events, company-specific facts)
    or can be evaluated using simple LLM knowledge (for general facts).

    Flow:
    1. Analyze statement with TemporalDetector to identify temporal/factual indicators
    2. If web search required: Use full FactCheckerPipeline with research
    3. If general knowledge: Use simple JudgeModule without research

    This addresses the core issue where recent temporal claims (e.g., December 2025
    events) cannot be verified from LLM training data alone and need real-time
    web verification.

    Attributes:
        temporal_detector: Lightweight classifier for routing decisions
        simple_judge: Fast judge for general knowledge claims
        fact_checker: Full pipeline with web search for temporal/recent claims
    """

    def __init__(
        self,
        max_judge_iterations: int = 3,
        max_page_visits: int = 3
    ):
        """Initialize the hybrid judge module.

        Args:
            max_judge_iterations: Max search iterations for FactCheckerPipeline
            max_page_visits: Max pages to visit per search query
        """
        super().__init__()

        # Lightweight detector using Predict (faster than ChainOfThought)
        self.temporal_detector = dspy.Predict(TemporalDetector)

        # Simple path: fast evaluation without web search
        self.simple_judge = JudgeModule()

        # Complex path: full fact-checking with web research
        self.fact_checker = FactCheckerPipeline(
            max_judge_iterations=max_judge_iterations,
            max_page_visits=max_page_visits
        )

    def forward(self, statement: str) -> dspy.Prediction:
        """Evaluate a statement using intelligent routing.

        Args:
            statement: The statement to evaluate.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
                - routing_decision: Which path was taken (web_search | simple_judge)
                - routing_reasoning: Why that path was chosen
        """
        # Step 1: Detect if web search is needed
        detection = self.temporal_detector(statement=statement)

        routing_decision = "web_search" if detection.requires_web_search else "simple_judge"
        routing_reasoning = detection.reasoning

        # Step 2: Route to appropriate evaluation path
        if detection.requires_web_search:
            # Use full fact-checking pipeline with web research
            print(f"🔍 [HYBRID ROUTER] Web search required: {routing_reasoning}")
            result = self.fact_checker(statement=statement)
        else:
            # Use simple judge without web search
            print(f"💡 [HYBRID ROUTER] Simple judge sufficient: {routing_reasoning}")
            result = self.simple_judge(statement=statement)

        # Step 3: Return unified prediction with routing metadata
        return dspy.Prediction(
            statement=statement,
            overall_verdict=result.overall_verdict,
            confidence=result.confidence,
            reasoning=result.reasoning,
            routing_decision=routing_decision,
            routing_reasoning=routing_reasoning,
            # Pass through additional fields if from FactCheckerPipeline
            claims=getattr(result, 'claims', None),
            claim_results=getattr(result, 'claim_results', None),
        )
