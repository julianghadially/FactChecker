"""Evidence-aware judge module - fact checker with web search and evidence retrieval."""

import dspy
from src.factchecker.signatures.evidence_aware_judge import EvidenceAwareJudge
from src.factchecker.modules.search_query_generator_module import SearchQueryGeneratorModule
from src.factchecker.modules.evidence_retriever_module import EvidenceRetrieverModule
from src.factchecker.modules.source_prioritization_module import SourcePrioritizationModule
from src.factchecker.modules.evidence_quality_assessor_module import EvidenceQualityAssessorModule


class JudgeModule(dspy.Module):
    """Evidence-aware fact checker with web search and evidence retrieval pipeline.

    This module implements a multi-stage fact-checking pipeline with adaptive search:
    1. SearchQueryGenerator: Generates 1-3 targeted search queries for the statement
    2. EvidenceRetriever: Searches the web and scrapes content to gather evidence
    2.25. SourcePrioritizer: Ranks sources by relevance to specific claims in the statement
    2.5. EvidenceQualityAssessor: Evaluates if evidence is sufficient; generates follow-up queries if needed
    3. EvidenceRetriever (follow-up): Re-runs with targeted queries if initial evidence is insufficient
    4. EvidenceAwareJudge: Evaluates the statement using the gathered evidence

    This adaptive architecture ensures the system retrieves targeted, relevant evidence
    for specific claims (like corporate agreements, technical specifications) rather than
    giving up when initial broad searches return off-topic results or failed scrapes.

    The source prioritization step ensures that when evidence must be truncated to fit
    context window limits, the most claim-relevant sources are included first, preventing
    loss of critical evidence.
    """

    def __init__(self, max_evidence_length: int = 15000):
        """Initialize the evidence-aware judge module pipeline.

        Args:
            max_evidence_length: Maximum total characters of evidence to use (default 15000).
        """
        super().__init__()
        self.query_generator = SearchQueryGeneratorModule()
        self.evidence_retriever = EvidenceRetrieverModule()
        self.source_prioritizer = SourcePrioritizationModule()
        self.quality_assessor = EvidenceQualityAssessorModule()
        self.judge = dspy.ChainOfThought(EvidenceAwareJudge)
        self.max_evidence_length = max_evidence_length

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

        # Stage 2: Retrieve evidence from web sources (returns structured source data)
        evidence_result = self.evidence_retriever(queries=query_result.queries)
        all_sources = list(evidence_result.sources)

        # Stage 2.25: Prioritize sources by relevance to specific claims
        prioritization_result = self.source_prioritizer(
            statement=statement,
            sources=all_sources
        )

        # Sort sources by relevance score (descending) and concatenate up to max length
        scored_sources = prioritization_result.scored_sources
        scored_sources.sort(key=lambda s: s.get('relevance_score', 0), reverse=True)

        # Concatenate top-ranked sources' markdown content up to max_evidence_length
        combined_evidence = self._concatenate_sources(scored_sources, self.max_evidence_length)

        # Stage 2.5: Assess evidence quality and generate follow-up queries if needed
        quality_result = self.quality_assessor(
            statement=statement,
            evidence=combined_evidence
        )

        # Stage 3: Follow-up evidence retrieval if initial evidence is insufficient
        if not quality_result.is_sufficient and quality_result.followup_queries:
            # Run follow-up queries through evidence retriever
            followup_evidence_result = self.evidence_retriever(queries=quality_result.followup_queries)

            # Prioritize follow-up sources
            followup_prioritization_result = self.source_prioritizer(
                statement=statement,
                sources=followup_evidence_result.sources
            )

            # Sort and concatenate follow-up sources
            followup_scored_sources = followup_prioritization_result.scored_sources
            followup_scored_sources.sort(key=lambda s: s.get('relevance_score', 0), reverse=True)
            followup_evidence = self._concatenate_sources(
                followup_scored_sources,
                self.max_evidence_length
            )

            # Append new evidence to existing evidence
            combined_evidence += "\n\n## Follow-up Evidence\n\n" + followup_evidence

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

    def _concatenate_sources(self, scored_sources: list[dict], max_length: int) -> str:
        """Concatenate sources in priority order up to max_length.

        Args:
            scored_sources: List of source dicts sorted by relevance (highest first).
            max_length: Maximum total characters of evidence.

        Returns:
            Combined markdown content with source attribution.
        """
        evidence_chunks = []
        total_length = 0

        for source in scored_sources:
            # Format source with attribution
            chunk = f"## Source: {source['title']}\nURL: {source['url']}\n\n{source['markdown']}\n\n---\n\n"

            # Check if adding this chunk would exceed limit
            if total_length + len(chunk) > max_length:
                # Try to fit partial content
                remaining_space = max_length - total_length
                if remaining_space > 200:  # Only include if we can fit meaningful content
                    truncated_chunk = chunk[:remaining_space] + "\n\n[Evidence truncated due to length...]"
                    evidence_chunks.append(truncated_chunk)
                break

            evidence_chunks.append(chunk)
            total_length += len(chunk)

        # If no evidence was gathered, provide informative message
        if not evidence_chunks:
            return "No evidence could be retrieved from web sources."

        return "".join(evidence_chunks)
