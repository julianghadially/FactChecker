"""Search query generator signature for creating targeted web search queries."""

from dspy import Signature, InputField, OutputField


class SearchQueryGenerator(Signature):
    """Generate targeted search queries to gather evidence for fact-checking a statement.

    Analyze the statement and generate 1-3 specific, diverse search queries that will
    help verify or refute the claims made. Queries should:
    - Target different aspects or components of the statement
    - Be specific enough to find relevant, authoritative sources
    - Include relevant keywords, dates, names, or specific claims
    - Avoid redundancy - each query should explore a different angle

    Example:
    Statement: "The Eiffel Tower is 330 meters tall and was completed in 1889."
    Good queries:
    - "Eiffel Tower official height meters"
    - "Eiffel Tower construction completion date 1889"
    - "Eiffel Tower exact measurements"
    """

    statement: str = InputField(desc="The statement to fact-check")

    reasoning: str = OutputField(desc="Explanation of the query strategy and what each query aims to verify")
    queries: list[str] = OutputField(desc="1-3 targeted search queries to gather evidence (list of strings)")
