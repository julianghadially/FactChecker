"""Research signature for generating search queries."""

import dspy
from dspy import Signature, InputField, OutputField


class ResearchSignature(Signature):
    """Generate targeted search queries for fact-checking a statement.

    Create 2-3 specific, targeted search queries that will help verify
    the factual claims in the statement. Queries should be diverse and
    cover different aspects or claims within the statement.
    """

    statement: str = InputField(
        desc="The statement to fact-check"
    )
    topic: str = InputField(
        desc="The topic or domain of the statement (optional, may be empty)"
    )

    search_queries: list[str] = OutputField(
        desc="2-3 targeted search queries to verify the statement"
    )
