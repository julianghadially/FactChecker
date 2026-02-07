"""Source prioritization module for ranking evidence sources by claim relevance."""

import dspy
from src.factchecker.signatures.source_prioritizer import SourcePrioritizer


class SourcePrioritizationModule(dspy.Module):
    """Module that prioritizes scraped sources by relevance to specific claims.

    Takes a statement and a list of scraped sources (with metadata and content) and uses
    an LLM to score each source's relevance to the specific claims in the statement.

    This enables intelligent evidence truncation: when evidence must be cut to fit context
    window limits, the most claim-relevant sources are prioritized, ensuring critical
    evidence (e.g., press releases with exact figures) isn't lost in favor of less
    relevant general content.

    This is inserted between evidence retrieval (stage 2) and quality assessment (stage 2.5)
    in the fact-checking pipeline, before evidence concatenation and truncation.
    """

    def __init__(self):
        """Initialize the source prioritization module."""
        super().__init__()
        self.prioritizer = dspy.ChainOfThought(SourcePrioritizer)

    def forward(self, statement: str, sources: list[dict]) -> dspy.Prediction:
        """Score sources by relevance to the statement's specific claims.

        Args:
            statement: The statement being fact-checked.
            sources: List of dicts with keys: url, title, markdown, success.

        Returns:
            dspy.Prediction with:
                - scored_sources: List of dicts (original sources + relevance_score field)
                - reasoning: Explanation of relevance scoring
        """
        # Filter to only successful scrapes with content
        valid_sources = [s for s in sources if s.get('success') and s.get('markdown')]

        if not valid_sources:
            # No valid sources to prioritize
            return dspy.Prediction(
                scored_sources=[],
                reasoning="No valid sources with content available for prioritization."
            )

        # Format sources for the LLM (title, URL, content preview)
        sources_info_parts = []
        for i, source in enumerate(valid_sources, 1):
            # Use first 500 chars of markdown as preview for relevance assessment
            content_preview = source['markdown'][:500]
            sources_info_parts.append(
                f"{i}. Title: {source['title']}\n"
                f"   URL: {source['url']}\n"
                f"   Content preview: {content_preview}...\n"
            )

        sources_info = "\n".join(sources_info_parts)

        # Get relevance scores from LLM
        result = self.prioritizer(statement=statement, sources_info=sources_info)

        # Attach scores to sources
        scored_sources = []
        relevance_scores = result.relevance_scores if hasattr(result, 'relevance_scores') else []

        # Handle case where LLM returns fewer scores than sources
        for i, source in enumerate(valid_sources):
            scored_source = source.copy()
            scored_source['relevance_score'] = relevance_scores[i] if i < len(relevance_scores) else 0.5
            scored_sources.append(scored_source)

        return dspy.Prediction(
            scored_sources=scored_sources,
            reasoning=result.reasoning if hasattr(result, 'reasoning') else "Relevance scoring completed."
        )
