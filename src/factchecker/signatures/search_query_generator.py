"""Search query generator signature for creating targeted web search queries."""

from dspy import Signature, InputField, OutputField


class SearchQueryGenerator(Signature):
    """Generate targeted search queries to gather evidence for fact-checking a statement.

    Analyze the statement and generate 1-3 specific, diverse search queries that will
    help verify or refute the claims made.

    If claim_types and search_strategy are provided, adapt queries accordingly:
    - For corporate_financial: Add site:sec.gov, include fiscal periods (Q1 2024, FY2023)
    - For corporate_announcement: Target press release sites (site:prnewswire.com), add "[company] press release"
    - For technical_specification: Include model numbers, add "specification" or "datasheet" keywords
    - For historical_event: Add specific dates and historical keywords

    Queries should:
    - Target different aspects or components of the statement
    - Be specific enough to find relevant, authoritative sources
    - Include relevant keywords, dates, names, or specific claims
    - Apply the search_strategy recommendations when provided
    - Avoid redundancy - each query should explore a different angle

    Example without claim types:
    Statement: "The Eiffel Tower is 330 meters tall and was completed in 1889."
    Good queries:
    - "Eiffel Tower official height meters"
    - "Eiffel Tower construction completion date 1889"
    - "Eiffel Tower exact measurements"

    Example with claim types:
    Statement: "Apple reported revenue of $90.8 billion in Q1 2024"
    Claim types: ["corporate_financial", "factual_quantitative"]
    Strategy: "Prioritize SEC filings and investor relations pages"
    Good queries:
    - "Apple Q1 2024 revenue earnings site:sec.gov"
    - "Apple first quarter 2024 financial results site:investor.apple.com"
    - "Apple fiscal Q1 2024 revenue 90 billion"
    """

    statement: str = InputField(desc="The statement to fact-check")
    claim_types: list[str] = InputField(
        default_factory=list,
        desc="Types of claim (e.g., corporate_financial, factual_quantitative). Optional.",
    )
    search_strategy: str = InputField(
        default="",
        desc="Search strategy recommendations from claim analysis. Optional.",
    )

    reasoning: str = OutputField(desc="Explanation of the query strategy and what each query aims to verify")
    queries: list[str] = OutputField(desc="1-3 targeted search queries to gather evidence (list of strings)")
