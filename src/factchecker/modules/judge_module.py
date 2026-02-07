"""Evidence-aware judge module - fact checker with web search and evidence retrieval."""

import dspy
from src.factchecker.signatures.evidence_aware_judge import EvidenceAwareJudge
from src.factchecker.modules.search_query_generator_module import SearchQueryGeneratorModule
from src.factchecker.modules.evidence_retriever_module import EvidenceRetrieverModule


class JudgeModule(dspy.Module):
    """Evidence-aware fact checker with two-phase search and evidence retrieval pipeline.

    This module implements a multi-stage fact-checking pipeline with prioritized search:
    1. SearchQueryGenerator: Generates two types of queries (primary source + general)
    2. EvidenceRetriever: Implements prioritized retrieval (primary sources first, then general)
    3. EvidenceAwareJudge: Evaluates the statement using the gathered evidence

    The two-phase approach prioritizes authoritative primary sources (official sites,
    index providers, government sites) before falling back to general searches,
    improving evidence quality and relevance.
    """

    def __init__(self):
        """Initialize the evidence-aware judge module pipeline."""
        super().__init__()
        self.query_generator = SearchQueryGeneratorModule()
        self.evidence_retriever = EvidenceRetrieverModule()
        self.judge = dspy.ChainOfThought(EvidenceAwareJudge)

    def forward(self, statement: str) -> dspy.Prediction:
        """Evaluate a statement for factual correctness using two-phase web evidence retrieval.

        Args:
            statement: The statement to evaluate.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict citing evidence
                - primary_source_queries: List of site-specific queries used (for transparency)
                - general_queries: List of general queries used (for transparency)
                - sources: List of source URLs consulted with query_type (for transparency)
        """
        # Stage 1: Generate search queries (two-phase strategy)
        query_result = self.query_generator(statement=statement)

        # Stage 2: Retrieve evidence from web sources (prioritized retrieval)
        evidence_result = self.evidence_retriever(
            primary_source_queries=query_result.primary_source_queries,
            general_queries=query_result.general_queries
        )

        # Stage 3: Judge the statement using evidence
        judgment = self.judge(statement=statement, evidence=evidence_result.evidence)

        # Return prediction with original format plus transparency fields
        return dspy.Prediction(
            statement=statement,
            overall_verdict=judgment.verdict,
            confidence=judgment.confidence,
            reasoning=judgment.reasoning,
            # Additional fields for transparency and debugging
            primary_source_queries=query_result.primary_source_queries,
            general_queries=query_result.general_queries,
            sources=evidence_result.sources,
        )
