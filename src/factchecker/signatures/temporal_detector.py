"""Temporal detector signature for identifying claims requiring web search."""

from dspy import Signature, InputField, OutputField


class TemporalDetector(Signature):
    """Detect whether a statement requires web-based verification.

    Analyze the statement to determine if it contains temporal indicators or
    specific factual details that require real-time verification:

    Web search is REQUIRED for:
    - Specific dates or recent time references (e.g., "December 2025", "last week", "recently")
    - Recent events or announcements (e.g., "announced", "plans to", "will launch")
    - Company-specific claims (SEC filings, board decisions, earnings, acquisitions)
    - Specific numerical data (financial figures, percentages, statistics with sources)
    - Current state claims that could change (e.g., "currently CEO of", "latest version")
    - Future claims or predictions (e.g., "will release", "expected to")

    Web search is NOT required for:
    - General knowledge facts (historical events, scientific principles, geography)
    - Well-established information (e.g., "Paris is capital of France")
    - Definitional or conceptual claims (e.g., "AI is machine learning")
    - Mathematical or logical statements that can be verified without external data
    """

    statement: str = InputField(desc="The statement to analyze for temporal/factual indicators")

    reasoning: str = OutputField(
        desc="Detailed explanation of why web search is or isn't needed. "
             "Identify specific temporal indicators, recent events, company claims, "
             "or factual details that require verification. Explain what type of "
             "information would need to be looked up."
    )
    requires_web_search: bool = OutputField(
        desc="True if the statement contains temporal indicators, recent events, "
             "company-specific claims, or specific factual details requiring verification. "
             "False if it's general knowledge that can be assessed from training data."
    )
