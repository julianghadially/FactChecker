"""Fire Judge module for iterative claim evaluation with web research."""

import dspy
from src.factchecker.signatures.fire_judge import FireJudge
from src.factchecker.models.data_types import JudgmentResult
from .research_agent_module import ResearchAgentModule
from .search_query_generator import SearchQueryGeneratorModule


class FireJudgeModule(dspy.Module):
    """FIRE (Fact-checking with Iterative Research and Evaluation) Judge.

    Iteratively evaluates a claim, requesting web searches as needed,
    until it reaches a verdict or exhausts the search budget.

    Attributes:
        max_iterations: Maximum number of search iterations allowed.
        research_agent: Module for conducting web research.
        query_generator: Module for generating initial search queries.
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
        self.query_generator = SearchQueryGeneratorModule()
        self.max_iterations = max_iterations

    def forward(self, claim: str) -> dspy.Prediction:
        """Evaluate a claim with iterative research.

        Args:
            claim: The factual claim to verify.

        Returns:
            JudgmentResult containing verdict, evidence, and metadata.
        """
        evidence = ""
        search_history: list[str] = []

        # Generate and execute initial search query to pre-populate evidence
        # This ensures every claim starts with relevant web evidence rather than
        # relying on the LLM's internal knowledge
        query_result = self.query_generator(claim=claim)
        initial_query = query_result.search_query

        if initial_query:
            search_history.append(initial_query)
            initial_evidence = self.research_agent(
                claim=claim,
                query=initial_query
            )
            evidence = f"--- Search: {initial_query} ---\n{initial_evidence}"

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
        return dspy.Prediction(
            claim=claim,
            verdict="not_supported",
            evidence_summary=evidence,
            search_queries=search_history,
            iterations=self.max_iterations
        )
