"""Query generator signature for intelligent web search query optimization."""

from dspy import Signature, InputField, OutputField


class QueryGenerator(Signature):
    """Generate optimized search queries from a statement for effective web research.

    Extract key entities (companies, people, dates, specific claims) and formulate
    targeted queries that will retrieve authoritative and relevant evidence.

    Output 1-3 optimized search queries that:
    - Focus on verifiable facts and entities rather than verbatim statements
    - Use domain-specific terminology (e.g., "Deutsche Bank 3M rating 2025")
    - Target authoritative sources (financial reports, official announcements, news)
    - Break complex statements into focused searchable components

    Example transformations:
    - "Deutsche Bank upgraded 3M to buy" → ["Deutsche Bank 3M rating 2025", "3M stock upgrade Deutsche Bank"]
    - "Apple released iPhone 15 in September 2023" → ["Apple iPhone 15 release date", "iPhone 15 launch September 2023"]
    - "Elon Musk acquired Twitter for $44 billion" → ["Twitter acquisition price Elon Musk", "Elon Musk Twitter $44 billion deal"]
    """

    statement: str = InputField(
        desc="The statement to generate optimized search queries for"
    )

    reasoning: str = OutputField(
        desc="Explanation of the query generation strategy and key entities identified"
    )
    queries: list[str] = OutputField(
        desc="1-3 optimized search queries for retrieving relevant evidence (as a Python list of strings)"
    )
