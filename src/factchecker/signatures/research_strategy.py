"""Research strategy signature with reasoning."""

import dspy
from dspy import Signature, InputField, OutputField


class ResearchStrategy(Signature):
    """Generate research strategy with search queries and reasoning.

    Create 2-3 specific, targeted search queries that will help verify
    the factual claims in the statement. Explain your research approach
    and why these particular queries were chosen.
    """

    statement: str = InputField(
        desc="The statement to fact-check"
    )
    topic: str = InputField(
        desc="The topic or domain of the statement (optional, may be empty)"
    )

    reasoning: str = OutputField(
        desc="Explanation of the research strategy and why these queries were chosen"
    )
    search_queries: list[str] = OutputField(
        desc="2-3 targeted search queries to verify the statement"
    )
