"""Claim type analyzer signature for classifying claims and suggesting search strategies."""

from typing import Literal
from dspy import Signature, InputField, OutputField


class ClaimTypeAnalyzer(Signature):
    """Analyze claim type and suggest targeted search strategy.

    Classify the claim into one or more categories to enable targeted
    evidence gathering. Consider:
    - What type of information is being claimed?
    - What sources would be most authoritative?
    - What search techniques would be most effective?

    Claim Types:
    - factual_quantitative: Numeric/measurable facts (height, date, revenue, statistics)
    - corporate_announcement: Company news, partnerships, product launches
    - corporate_financial: Financial metrics, earnings, stock data, analyst ratings
    - historical_event: Past events with specific dates or time periods
    - general_knowledge: Common facts, definitions, well-known information
    - current_event: Recent news within the last 6 months
    - technical_specification: Product specs, technical details, industry standards

    Search Strategy should provide specific, actionable guidance such as:
    - Which domains to prioritize (e.g., "site:sec.gov for financial data")
    - What keywords to include (e.g., "10-K", "press release", "Q1 2024")
    - Date context to add (e.g., "fiscal year 2023", "quarter")
    - Authoritative source types (e.g., "official documentation", "investor relations")

    Example:
    Statement: "Apple reported revenue of $90.8 billion in Q1 2024"
    Claim types: ["corporate_financial", "factual_quantitative"]
    Strategy: "Prioritize SEC filings (site:sec.gov) and investor relations pages (site:investor.apple.com). Include fiscal quarter terminology (Q1 2024, first quarter 2024) and specific revenue figures in queries."
    """

    statement: str = InputField(desc="The claim to analyze")

    reasoning: str = OutputField(
        desc="Analysis of the claim's characteristics and why these types and strategy apply"
    )
    claim_types: list[
        Literal[
            "factual_quantitative",
            "corporate_announcement",
            "corporate_financial",
            "historical_event",
            "general_knowledge",
            "current_event",
            "technical_specification",
        ]
    ] = OutputField(
        desc="One or more claim types that apply (multiple types can apply to one claim)"
    )
    search_strategy: str = OutputField(
        desc="Specific search strategy recommendations including domains, keywords, and techniques"
    )
