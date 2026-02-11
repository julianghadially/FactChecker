"""Research signature for generating search queries from a statement."""

from dspy import Signature, InputField, OutputField


class Research(Signature):
    """Generate targeted search queries to verify a factual statement.

    Analyze the statement and topic to produce 2-3 specific search queries
    that would help gather evidence to verify or refute the claims made.
    Each query should target different aspects or angles of the statement.
    """

    statement: str = InputField(
        desc="The factual statement to research and verify"
    )
    topic: str = InputField(
        desc="The general topic or domain of the statement"
    )

    reasoning: str = OutputField(
        desc="Explanation of the research strategy and why these queries were chosen"
    )
    search_queries: list[str] = OutputField(
        desc="2-3 targeted search queries to gather evidence about the statement"
    )
