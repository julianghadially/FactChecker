"""Search Query Generator signature for extracting key elements from claims."""

from dspy import Signature, InputField, OutputField


class SearchQueryGenerator(Signature):
    """Generate an optimized web search query from a factual claim.

    Analyzes the claim to identify:
    1. Named entities (companies, people, organizations, locations)
    2. Specific metrics, numbers, percentages, or quantities
    3. Time periods (dates, years, timeframes)
    4. Action verbs and key relationships

    Outputs a search query optimized for web search engines, using:
    - Quotes around exact phrases that should be matched verbatim
    - Key terms that maximize search relevance
    - Entity names and specific metrics for precision
    """

    claim: str = InputField(desc="A single factual claim to generate a search query for")
    reasoning: str = OutputField(
        desc="Step-by-step analysis explaining: (1) identified named entities, "
             "(2) specific metrics/numbers/percentages, (3) time periods, "
             "(4) action verbs and key relationships"
    )
    search_query: str = OutputField(
        desc="Optimized search query for web search engines. Use quotes around "
             "exact phrases and entity names. Include key metrics and time periods. "
             "Should be concise but specific enough to find relevant evidence."
    )
