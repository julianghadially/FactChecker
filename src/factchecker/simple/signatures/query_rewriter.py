"""Query rewriter signature for optimizing search queries from statements."""

from dspy import Signature, InputField, OutputField


class QueryRewriter(Signature):
    """Transform fact-checking statements into optimized search queries.

    Detects negative assertions (e.g., "has not", "did not", "never", "no evidence")
    and rewrites them as positive searches focusing on what would disprove the
    negative claim. For example:
    - "U.S. Bancorp has not paid dividends in 2025" → "U.S. Bancorp dividends paid 2025"
    - "There is no evidence of X" → "X evidence"
    - "Company Y never released product Z" → "Company Y product Z release"

    For non-negative statements, simplifies into a factual search query by extracting
    key entities and removing opinion words.
    """

    statement: str = InputField(
        desc="The original statement to convert into a search query"
    )

    search_query: str = OutputField(
        desc="Optimized search query. For negative claims, rewrite as positive query "
             "focusing on what would disprove the claim. For normal statements, "
             "extract key factual terms. Keep it concise (under 10 words)."
    )
