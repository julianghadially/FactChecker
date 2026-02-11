"""FactChecker pipeline combining research and judgment stages."""

import dspy
from src.factchecker.modules.research_module import ResearchModule
from src.factchecker.modules.judge_module import JudgeModule


class FactCheckerPipeline(dspy.Module):
    """Complete fact-checking pipeline with research and judgment stages.

    This pipeline orchestrates the two-stage fact-checking process:
    1. ResearchModule: Generates search queries to gather evidence
    2. JudgeModule: Evaluates the statement's factual correctness

    Current implementation: ResearchModule generates queries but evidence
    gathering is a placeholder. JudgeModule makes judgments based on LLM
    knowledge without external evidence.

    Future implementation: ResearchModule will execute searches via SERPER,
    scrape content via Firecrawl, and pass gathered evidence to an enhanced
    judge that can evaluate based on external sources.
    """

    def __init__(self):
        """Initialize the fact-checking pipeline."""
        super().__init__()
        self.research = ResearchModule()
        self.judge = JudgeModule()

    def forward(self, statement: str, topic: str = "general") -> dspy.Prediction:
        """Execute the complete fact-checking pipeline.

        Args:
            statement: The statement to fact-check.
            topic: The topic/domain of the statement (default: "general").

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - topic: The input topic
                - queries: Research queries generated
                - research_reasoning: Research strategy explanation
                - evidence_summary: Placeholder evidence summary
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - judgment_reasoning: Judge's explanation of verdict
        """
        # Stage 1: Research - generate search queries
        research_result = self.research(statement=statement, topic=topic)

        # Stage 2: Judge - evaluate statement
        # Note: Current judge doesn't use research evidence
        # Future: pass evidence_summary to enhanced judge signature
        judge_result = self.judge(statement=statement)

        # Combine results into unified prediction
        return dspy.Prediction(
            statement=statement,
            topic=topic,
            queries=research_result.queries,
            research_reasoning=research_result.reasoning,
            evidence_summary=research_result.evidence_summary,
            overall_verdict=judge_result.overall_verdict,
            confidence=judge_result.confidence,
            judgment_reasoning=judge_result.reasoning
        )
