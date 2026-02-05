"""Judge module - fact checker with optional web research."""

import dspy
from src.factchecker.signatures.judge import Judge
from src.factchecker.modules.research_module import ResearchModule


class JudgeModule(dspy.Module):
    """Fact checker that judges statements with optional web research.

    Takes a statement as input and outputs a verdict. When research is enabled,
    it first retrieves web evidence before making a judgment.

    This module can operate in two modes:
    - With research (default): Searches web and scrapes sources for evidence
    - Without research: Uses only LLM knowledge (faster but less accurate)
    """

    def __init__(self, use_research: bool = True):
        """Initialize the judge module.

        Args:
            use_research: Whether to use web research before judging (default: True).
        """
        super().__init__()
        self.judge = dspy.ChainOfThought(Judge)
        self.use_research = use_research
        if use_research:
            self.research_module = ResearchModule(num_queries=2, num_sources=5)

    def forward(self, statement: str) -> dspy.Prediction:
        """Evaluate a statement for factual correctness with optional web research.

        Args:
            statement: The statement to evaluate.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict
                - evidence: Summary of web evidence (if research enabled)
                - sources: List of source URLs and metadata (if research enabled)
        """
        # Step 1: Perform web research if enabled
        evidence = ""
        sources = []
        if self.use_research:
            research_result = self.research_module(statement=statement)
            if research_result.success:
                evidence = research_result.evidence_summary
                sources = research_result.sources

        # Step 2: Judge with evidence
        result = self.judge(statement=statement, evidence=evidence)

        # Step 3: Return enhanced prediction
        return dspy.Prediction(
            statement=statement,
            overall_verdict=result.verdict,
            confidence=result.confidence,
            reasoning=result.reasoning,
            evidence=evidence,
            sources=sources,
        )
