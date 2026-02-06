"""FactChecker pipeline combining research and judgment."""

import dspy
from src.factchecker.modules.research_module import ResearchModule
from src.factchecker.modules.judge_module import JudgeModule


class FactCheckerPipeline(dspy.Module):
    """Research-enhanced fact-checking pipeline.

    Combines ResearchModule (query generation) with JudgeModule (evaluation)
    to provide both research context and factual judgment. Currently the
    modules operate independently, but future versions will pass evidence
    from research to the judge.

    This pipeline is designed to be optimizable with DSPy's optimization
    techniques, allowing both research and judgment strategies to be tuned.
    """

    def __init__(self):
        """Initialize the pipeline with research and judge modules."""
        super().__init__()
        self.research = ResearchModule()
        self.judge = JudgeModule()

    def forward(self, statement: str, topic: str = "") -> dspy.Prediction:
        """Fact-check a statement with research-enhanced context.

        Args:
            statement: The statement to fact-check.
            topic: Optional topic/domain context for the statement.

        Returns:
            dspy.Prediction with:
                - statement: Original input statement
                - search_queries: Generated search queries (list[str])
                - evidence_summary: Placeholder evidence summary
                - research_reasoning: Research strategy explanation
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Confidence score (0.0-1.0)
                - reasoning: Judgment reasoning
        """
        # Step 1: Generate research queries
        research_result = self.research(statement=statement, topic=topic)

        # Step 2: Judge the statement
        # Note: Currently judge doesn't use evidence, but will in future versions
        judgment = self.judge(statement=statement)

        # Step 3: Combine results
        return dspy.Prediction(
            statement=statement,
            search_queries=research_result.queries,
            evidence_summary=research_result.evidence_summary,
            research_reasoning=research_result.reasoning,
            overall_verdict=judgment.overall_verdict,
            confidence=judgment.confidence,
            reasoning=judgment.reasoning
        )
