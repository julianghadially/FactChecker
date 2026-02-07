"""Source prioritizer signature for ranking evidence sources by relevance."""

from dspy import Signature, InputField, OutputField


class SourcePrioritizer(Signature):
    """Prioritize scraped web sources by relevance to specific claims in the statement.

    Given a statement to fact-check and a list of scraped sources (with titles and URLs),
    analyze each source's relevance to the SPECIFIC CLAIMS made in the statement.

    This prioritization ensures that when evidence must be truncated due to context window
    limits, the most claim-relevant sources are included first, preventing loss of critical
    evidence (e.g., a BusinessWire article with exact figures mentioned in the statement).

    Relevance scoring should consider:
    1. Direct mention of key entities, names, companies, or products from the statement
    2. Presence of specific figures, dates, or technical details claimed in the statement
    3. Topical alignment with the core claims (not just general topic area)
    4. Authority and specificity (e.g., press releases > general news > blogs for corporate claims)

    For example:
    - Statement: "Company X raised $15 million in Series A funding in 2023"
    - High relevance (0.9-1.0): Press release announcing "Company X raises $15M Series A"
    - Medium relevance (0.5-0.7): General article about Company X mentioning funding
    - Low relevance (0.1-0.3): Article about Company X's product with no funding details
    - Very low relevance (0.0-0.1): Industry news about similar companies' funding

    Be precise: a source must directly address the specific claims to score highly.
    """

    statement: str = InputField(desc="The statement being fact-checked, containing specific claims to verify")
    sources_info: str = InputField(
        desc="List of scraped sources with their metadata (title, URL, preview of content). Format: numbered list with title, URL, and content preview for each source."
    )

    reasoning: str = OutputField(
        desc="Brief explanation of relevance assessment for each source, identifying which specific claims from the statement each source addresses (or doesn't address)"
    )
    relevance_scores: list[float] = OutputField(
        desc="List of relevance scores (0.0 to 1.0) for each source, in the same order as sources_info. Higher scores = more directly relevant to specific claims in the statement."
    )
