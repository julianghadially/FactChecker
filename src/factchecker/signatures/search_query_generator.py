"""Search query generator signature for creating targeted web search queries."""

from dspy import Signature, InputField, OutputField


class SearchQueryGenerator(Signature):
    """Generate targeted search queries using a two-phase strategy for fact-checking.

    Analyze the statement and generate TWO types of queries:

    1. PRIMARY SOURCE QUERIES (1-2 queries): Site-specific queries targeting authoritative
       primary sources using the site: operator. These should target:
       - Official organization websites (e.g., company investor relations, government sites)
       - Index providers and financial institutions (e.g., S&P Global, MSCI)
       - Industry authorities and regulatory bodies
       - Official biographies and institutional pages
       Example: "site:spglobal.com S&P 500 Dividend Aristocrats constituents list Caterpillar"

    2. GENERAL QUERIES (1-2 queries): Broader queries for context and verification.
       These provide supporting evidence and cross-validation from news, analysis, etc.
       Example: "Caterpillar S&P 500 Dividend Aristocrats member history"

    Example:
    Statement: "Caterpillar is part of the S&P 500 Dividend Aristocrats index."
    Good queries:
    Primary source: ["site:spglobal.com S&P 500 Dividend Aristocrats constituents list Caterpillar"]
    General: ["Caterpillar S&P 500 Dividend Aristocrats member history"]
    """

    statement: str = InputField(desc="The statement to fact-check")

    reasoning: str = OutputField(desc="Explanation of the two-phase query strategy and what each query aims to verify")
    primary_source_queries: list[str] = OutputField(desc="1-2 site-specific queries targeting authoritative primary sources using site: operator (list of strings)")
    general_queries: list[str] = OutputField(desc="1-2 broader queries for context and verification (list of strings)")
