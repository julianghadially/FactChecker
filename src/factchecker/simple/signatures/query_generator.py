"""Query generator signature for multi-query search strategy."""

from dspy import Signature, InputField, OutputField


class QueryGenerator(Signature):
    """Extract focused search queries from a statement to gather comprehensive evidence.

    Analyze the statement and generate 1-3 specific, targeted search queries that would
    help verify different aspects of the claim, especially temporal and numeric details.
    Each query should be concise and optimized for web search.

    Examples:
    - Statement: "Mondelez has been selling sugar-free Oreo cookies in the United States
      for several years prior to the announced Oreo Zero Sugar launch"
      Queries: ["Oreo Zero Sugar launch date", "sugar-free Oreo United States history
      before 2026", "Mondelez sugar-free Oreo products US availability"]

    - Statement: "Apple released the iPhone 15 in September 2023"
      Queries: ["iPhone 15 release date", "Apple iPhone 15 launch September 2023"]
    """

    statement: str = InputField(desc="The statement to generate search queries for")

    queries: list[str] = OutputField(
        desc="1-3 focused search queries that target different aspects of the statement, "
             "especially temporal and numeric claims. Each query should be concise (5-15 words)."
    )
