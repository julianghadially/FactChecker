"""FactChecker pipeline that combines research and judgment."""

import dspy
from src.factchecker.modules.research_module import ResearchModule
from src.factchecker.modules.judge_module import JudgeModule


class FactCheckerPipeline(dspy.Module):
    """Complete fact-checking pipeline with research and judgment phases.

    This pipeline orchestrates two main phases:
    1. Research Phase: Generate targeted search queries (ResearchModule)
    2. Judgment Phase: Evaluate statement factual correctness (JudgeModule)

    Currently, the judgment phase does not use the research evidence.
    Future versions will pass evidence_summary to an enhanced judge module
    that can ground its verdict in the gathered evidence.
    """

    def __init__(self):
        """Initialize the fact-checker pipeline."""
        super().__init__()
        self.research = ResearchModule()
        self.judge = JudgeModule()

    def forward(self, statement: str, topic: str) -> dspy.Prediction:
        """Execute full fact-checking pipeline.

        Args:
            statement: The statement to fact-check.
            topic: The topic/domain of the statement.

        Returns:
            dspy.Prediction with:
                - statement: Original input statement
                - topic: Original input topic
                - queries: List of search queries generated
                - evidence_summary: Placeholder research summary
                - overall_verdict: Factual correctness verdict
                - confidence: Confidence score (0.0 to 1.0)
                - reasoning: Explanation of the verdict
        """
        # Phase 1: Research
        research_result = self.research(statement=statement, topic=topic)

        # Phase 2: Judgment (currently independent of research)
        judge_result = self.judge(statement=statement)

        # Combine results into unified prediction
        return dspy.Prediction(
            statement=statement,
            topic=topic,
            queries=research_result.queries,
            evidence_summary=research_result.evidence_summary,
            overall_verdict=judge_result.overall_verdict,
            confidence=judge_result.confidence,
            reasoning=judge_result.reasoning
        )
