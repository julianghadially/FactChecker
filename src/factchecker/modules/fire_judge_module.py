"""Fire Judge module for iterative claim evaluation with web research."""

import re
import dspy
from src.factchecker.signatures.fire_judge import FireJudge
from src.factchecker.models.data_types import JudgmentResult
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

    def _detect_temporal_markers(self, claim: str) -> bool:
        """Detect if the claim contains temporal markers suggesting time-sensitive content.

        Args:
            claim: The claim text to analyze.

        Returns:
            True if temporal markers are detected, False otherwise.
        """
        claim_lower = claim.lower()

        # Detect years 2024 and beyond
        year_pattern = r'\b(202[4-9]|20[3-9]\d)\b'
        if re.search(year_pattern, claim):
            return True

        # Detect month names
        months = [
            'january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december',
            'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
        ]
        if any(month in claim_lower for month in months):
            return True

        # Detect temporal keywords
        temporal_keywords = [
            'recent', 'recently', 'latest', 'current', 'currently', 'now',
            'today', 'this year', 'this month', 'this week', 'upcoming',
            'announced', 'will', 'plans to', 'scheduled', 'ongoing'
        ]
        if any(keyword in claim_lower for keyword in temporal_keywords):
            return True

        return False

    def forward(self, claim: str) -> dspy.Prediction:
        """Evaluate a claim with iterative research.

        Args:
            claim: The factual claim to verify.

        Returns:
            JudgmentResult containing verdict, evidence, and metadata.
        """
        evidence = ""
        search_history: list[str] = []

        # Pre-processing: automatically trigger initial search for temporal claims
        # or when no evidence is available on first iteration
        has_temporal_markers = self._detect_temporal_markers(claim)
        if has_temporal_markers or not evidence:
            # Generate initial research query
            initial_query = f"{claim} verification"
            search_history.append(initial_query)

            # Execute initial research
            new_evidence = self.research_agent(
                claim=claim,
                query=initial_query
            )
            evidence += f"--- Search: {initial_query} ---\n{new_evidence}"

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
