"""Fact checker pipeline module orchestrating the complete fact-checking flow."""

import dspy
from src.factchecker.models.data_types import FactCheckResult
from .claim_extractor_module import ClaimExtractorModule
from .fire_judge_module import FireJudgeModule
from .research_agent_module import ResearchAgentModule
from .aggregator_module import AggregatorModule
from src.services.serper_service import SerperService
from src.services.firecrawl_service import FirecrawlService


class FactCheckerPipeline(dspy.Module):
    """Complete fact-checking pipeline that orchestrates all modules.

    Flow:
    1. Extract claims from statement
    2. Evaluate each claim with iterative web research
    3. Aggregate verdicts into overall statement verdict

    Attributes:
        claim_extractor: Module for extracting claims.
        fire_judge: Module for evaluating claims with research.
        aggregator: Module for aggregating verdicts.
    """

    def __init__(
        self,
        max_judge_iterations: int = 3,
        max_page_visits: int = 3
    ):
        """Initialize the fact checker pipeline.

        Args:
            serper_api_key: API key for Serper web search.
            firecrawl_api_key: API key for Firecrawl page scraping.
            max_judge_iterations: Max search iterations per claim.
            max_page_visits: Max pages to visit per search query.
        """
        super().__init__()

        
        # Initialize modules
        self.claim_extractor = ClaimExtractorModule()
        self.research_agent = ResearchAgentModule(
            max_page_visits=max_page_visits
        )
        self.fire_judge = FireJudgeModule(
            self.research_agent,
            max_judge_iterations
        )
        self.aggregator = AggregatorModule()

    def forward(self, statement: str) -> dspy.Prediction:
        """Execute the full fact-checking pipeline.

        Args:
            statement: The statement to fact-check.

        Returns:
            FactCheckResult with all details including claim-level results.
        """
        # Step 1: Extract claims
        claims = self.claim_extractor(statement=statement)

        # Step 2: Evaluate each claim
        claim_results = []
        for claim in claims:
            result = self.fire_judge(claim=claim)
            claim_results.append(result)

        # Step 3: Aggregate verdicts
        claim_verdicts = [
            {
                "claim": r.claim,
                "verdict": r.verdict,
                "evidence_summary": r.evidence_summary
            }
            for r in claim_results
        ]

        aggregation = self.aggregator(
            original_statement=statement,
            claim_verdicts=claim_verdicts
        )

        return dspy.Prediction(
            statement=statement,
            claims=claims,
            claim_results=claim_results,
            overall_verdict=aggregation.verdict,
            confidence=aggregation.confidence,
            reasoning=aggregation.reasoning
        )
