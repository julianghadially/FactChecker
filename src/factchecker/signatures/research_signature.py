"""Research signature for generating targeted search queries."""

from dspy import Signature, InputField, OutputField
from typing import List


class ResearchSignature(Signature):
    """Generate targeted search queries for fact-checking a statement.

    Given a statement and topic, produce 2-3 specific search queries that
    would help verify or refute the claims in the statement. Queries should
    be focused, diverse, and designed to surface relevant evidence.
    """

    statement: str = InputField(
        desc="The statement to fact-check"
    )
    topic: str = InputField(
        desc="The general topic or domain of the statement (e.g., 'politics', 'science', 'technology')"
    )

    reasoning: str = OutputField(
        desc="Explanation of the query generation strategy and what each query aims to verify"
    )
    search_queries: List[str] = OutputField(
        desc="2-3 targeted search queries to gather evidence. Each query should be specific and designed to find factual information"
    )
