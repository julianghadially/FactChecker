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

    def forward(self, statement: str, url: str = None) -> dspy.Prediction:
        """Evaluate a statement for factual correctness using web evidence.

        Args:
            statement: The statement to evaluate.
            url: Optional comma-separated string of URLs to scrape directly for evidence.
                 These URLs will be scraped first before search-based evidence gathering.

        Returns:
            dspy.Prediction with:
                - statement: The input statement
                - overall_verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
                - confidence: Float between 0.0 and 1.0
                - reasoning: Explanation of the verdict citing evidence
                - queries: List of search queries used (for transparency)
                - sources: List of source URLs consulted (for transparency)
        """
        # Stage 1: Generate search queries (always done as backup)
        query_result = self.query_generator(statement=statement)

        # Stage 2a: If URLs provided, retrieve evidence directly from them first
        all_evidence_parts = []
        all_sources = []

        if url:
            # Parse comma-separated URLs
            url_list = [u.strip() for u in url.split(',') if u.strip()]

            if url_list:
                # Retrieve evidence from provided URLs
                url_evidence_result = self.evidence_retriever.forward_from_urls(urls=url_list)

                # Add URL-based evidence with clear marker
                if url_evidence_result.evidence.strip() and url_evidence_result.evidence != "No evidence could be retrieved from provided URLs.":
                    all_evidence_parts.append("=== EVIDENCE FROM PROVIDED AUTHORITATIVE SOURCES ===\n\n")
                    all_evidence_parts.append(url_evidence_result.evidence)
                    all_sources.extend(url_evidence_result.sources)

        # Stage 2b: Retrieve evidence from search-based sources (as backup/additional context)
        search_evidence_result = self.evidence_retriever(queries=query_result.queries)

        # Add search-based evidence with clear separation
        if search_evidence_result.evidence.strip() and search_evidence_result.evidence != "No evidence could be retrieved from web sources.":
            if all_evidence_parts:
                all_evidence_parts.append("\n\n=== ADDITIONAL EVIDENCE FROM WEB SEARCH ===\n\n")
            all_evidence_parts.append(search_evidence_result.evidence)

        all_sources.extend(search_evidence_result.sources)

        # Combine all evidence
        combined_evidence = "".join(all_evidence_parts) if all_evidence_parts else "No evidence could be retrieved from any sources."

        # Stage 3: Judge the statement using combined evidence
        judgment = self.judge(statement=statement, evidence=combined_evidence)

        # Return prediction with original format plus transparency fields
        return dspy.Prediction(
            statement=statement,
            overall_verdict=judgment.verdict,
            confidence=judgment.confidence,
            reasoning=judgment.reasoning,
            # Additional fields for transparency and debugging
            queries=query_result.queries,
            sources=all_sources,
        )
