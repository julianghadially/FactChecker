"""Evidence-aware judge module - fact checker with web search and evidence retrieval."""

import dspy
from src.factchecker.signatures.evidence_aware_judge import EvidenceAwareJudge
from src.factchecker.modules.search_query_generator_module import SearchQueryGeneratorModule
from src.factchecker.modules.evidence_retriever_module import EvidenceRetrieverModule
from src.factchecker.modules.source_deep_dive_module import SourceDeepDiveModule
from src.factchecker.modules.evidence_quality_assessor_module import EvidenceQualityAssessorModule


class JudgeModule(dspy.Module):
    """Evidence-aware fact checker with web search and evidence retrieval pipeline.

    This module implements a multi-stage fact-checking pipeline with adaptive search:
    1. SearchQueryGenerator: Generates 1-3 targeted search queries for the statement
    2. EvidenceRetriever: Searches the web and scrapes content to gather evidence
    2.25. SourceDeepDive: Analyzes initial evidence and generates site-specific queries for deeper investigation
    2.5. EvidenceQualityAssessor: Evaluates if evidence is sufficient; generates follow-up queries if needed
    3. EvidenceRetriever (follow-up): Re-runs with targeted queries if initial evidence is insufficient
    4. EvidenceAwareJudge: Evaluates the statement using the gathered evidence

    This adaptive architecture ensures the system retrieves targeted, relevant evidence
    for specific claims (like corporate agreements, technical specifications) rather than
    giving up when initial broad searches return off-topic results or failed scrapes.

    The SourceDeepDive stage enables multi-hop reasoning by identifying authoritative sources
    in the initial evidence and drilling deeper with site-specific queries (e.g., site:tesu.edu)
    to discover program-level details and context that may not appear in broad searches.
    """

    def __init__(self):
        """Initialize the evidence-aware judge module pipeline."""
        super().__init__()
        self.query_generator = SearchQueryGeneratorModule()
        self.evidence_retriever = EvidenceRetrieverModule()
        self.source_deep_dive = SourceDeepDiveModule()
        self.quality_assessor = EvidenceQualityAssessorModule()
        self.judge = dspy.ChainOfThought(EvidenceAwareJudge)

    def forward(self, statement: str) -> dspy.Prediction:
        """Evaluate a statement for factual correctness using web evidence with adaptive search.

        Args:
            statement: The statement to evaluate.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict citing evidence
                - queries: List of all search queries used, including follow-ups (for transparency)
                - sources: List of source URLs consulted (for transparency)
                - quality_assessment: Assessment of evidence quality (for debugging)
        """
        # Stage 1: Generate initial search queries
        query_result = self.query_generator(statement=statement)
        all_queries = list(query_result.queries)

        # Stage 2: Retrieve evidence from web sources
        evidence_result = self.evidence_retriever(queries=query_result.queries)
        combined_evidence = evidence_result.evidence
        all_sources = list(evidence_result.sources)

        # Stage 2.25: Deep dive into promising sources with site-specific queries
        source_deep_dive_result = self.source_deep_dive(
            statement=statement,
            evidence=combined_evidence
        )

        # Retrieve additional evidence from deep dive queries if any were generated
        if source_deep_dive_result.targeted_site_queries:
            deep_dive_evidence_result = self.evidence_retriever(
                queries=source_deep_dive_result.targeted_site_queries
            )

            # Append deep dive evidence to combined evidence
            combined_evidence += "\n\n## Deep Dive Evidence\n\n" + deep_dive_evidence_result.evidence

            # Track deep dive queries and sources
            all_queries.extend(source_deep_dive_result.targeted_site_queries)
            all_sources.extend(deep_dive_evidence_result.sources)

        # Stage 2.5: Assess evidence quality and generate follow-up queries if needed
        quality_result = self.quality_assessor(
            statement=statement,
            evidence=combined_evidence
        )

        # Stage 3: Follow-up evidence retrieval if initial evidence is insufficient
        if not quality_result.is_sufficient and quality_result.followup_queries:
            # Run follow-up queries through evidence retriever
            followup_evidence_result = self.evidence_retriever(queries=quality_result.followup_queries)

            # Append new evidence to existing evidence
            combined_evidence += "\n\n## Follow-up Evidence\n\n" + followup_evidence_result.evidence

            # Track all queries and sources
            all_queries.extend(quality_result.followup_queries)
            all_sources.extend(followup_evidence_result.sources)

        # Stage 4: Judge the statement using all gathered evidence
        judgment = self.judge(statement=statement, evidence=combined_evidence)

        # Return prediction with original format plus transparency fields
        return dspy.Prediction(
            statement=statement,
            overall_verdict=judgment.verdict,
            confidence=judgment.confidence,
            reasoning=judgment.reasoning,
            # Additional fields for transparency and debugging
            queries=all_queries,
            sources=all_sources,
            quality_assessment=quality_result.quality_assessment,
        )
