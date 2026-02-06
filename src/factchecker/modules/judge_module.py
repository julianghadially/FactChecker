"""Evidence-aware judge module - fact checker with web search and evidence retrieval."""

import dspy
from src.factchecker.signatures.evidence_aware_judge import EvidenceAwareJudge
from src.factchecker.modules.search_query_generator_module import SearchQueryGeneratorModule
from src.factchecker.modules.evidence_retriever_module import EvidenceRetrieverModule


class JudgeModule(dspy.Module):
    """Evidence-aware fact checker with web search and evidence retrieval pipeline.

    This module implements a multi-stage fact-checking pipeline:
    1. SearchQueryGenerator: Generates 1-3 targeted search queries for the statement
    2. EvidenceRetriever: Searches the web and scrapes content to gather evidence
    3. EvidenceAwareJudge: Evaluates the statement using the gathered evidence

    This allows the system to verify recent events and specific claims beyond
    the LLM's knowledge cutoff by consulting authoritative web sources.
    """

    def __init__(self):
        """Initialize the evidence-aware judge module pipeline."""
        super().__init__()
        self.query_generator = SearchQueryGeneratorModule()
        self.evidence_retriever = EvidenceRetrieverModule()
        self.judge = dspy.ChainOfThought(EvidenceAwareJudge)

    def forward(self, statement: str) -> dspy.Prediction:
        """Evaluate a statement for factual correctness using web evidence.

        Args:
            statement: The statement to evaluate.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict citing evidence
                - queries: List of search queries used (for transparency)
                - sources: List of source URLs consulted (for transparency)
        """
        # Stage 1: Generate search queries
        query_result = self.query_generator(statement=statement)

        # Stage 2: Retrieve evidence from web sources
        evidence_result = self.evidence_retriever(queries=query_result.queries)

        # Stage 3: Judge the statement using evidence
        judgment = self.judge(statement=statement, evidence=evidence_result.evidence)

        # Return prediction with original format plus transparency fields
        return dspy.Prediction(
            statement=statement,
            overall_verdict=judgment.verdict,
            confidence=judgment.confidence,
            reasoning=judgment.reasoning,
            # Additional fields for transparency and debugging
            queries=query_result.queries,
            sources=evidence_result.sources,
        )
