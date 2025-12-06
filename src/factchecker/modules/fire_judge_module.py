"""Fire Judge module for iterative claim evaluation with web research."""

import dspy
from ..signatures.fire_judge import FireJudge
from ..models.data_types import JudgmentResult
from .research_agent_module import ResearchAgentModule


class FireJudgeModule(dspy.Module):
    """FIRE (Fact-checking with Iterative Research and Evaluation) Judge.

    Iteratively evaluates a claim, requesting web searches as needed,
    until it reaches a verdict or exhausts the search budget.

    Attributes:
        max_iterations: Maximum number of search iterations allowed.
        research_agent: Module for conducting web research.
    """

    def __init__(
        self,
        research_agent: ResearchAgentModule,
        max_iterations: int = 3
    ):
        """Initialize the Fire Judge module.

        Args:
            research_agent: Module for conducting web research.
            max_iterations: Maximum search iterations before defaulting verdict.
        """
        super().__init__()
        self.judge = dspy.ChainOfThought(FireJudge)
        self.research_agent = research_agent
        self.max_iterations = max_iterations

    def forward(self, claim: str) -> JudgmentResult:
        """Evaluate a claim with iterative research.

        Args:
            claim: The factual claim to verify.

        Returns:
            JudgmentResult containing verdict, evidence, and metadata.
        """
        evidence = ""
        search_history: list[str] = []

        for iteration in range(self.max_iterations):
            result = self.judge(
                claim=claim,
                evidence=evidence,
                search_history=search_history
            )

            # If we have a verdict, return it
            if result.verdict:
                return JudgmentResult(
                    claim=claim,
                    verdict=result.verdict,
                    evidence_summary=evidence,
                    search_queries=search_history,
                    iterations=iteration + 1
                )

            # If we need more research and have a new query
            if result.next_search and result.next_search not in search_history:
                search_history.append(result.next_search)
                new_evidence = self.research_agent(
                    claim=claim,
                    query=result.next_search
                )
                evidence += f"\n\n--- Search: {result.next_search} ---\n{new_evidence}"

        # Exhausted iterations without verdict - default to not_supported
        return JudgmentResult(
            claim=claim,
            verdict="not_supported",
            evidence_summary=evidence,
            search_queries=search_history,
            iterations=self.max_iterations
        )
