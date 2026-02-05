"""Research signatures for web-based evidence retrieval."""

import dspy
from dspy import InputField, OutputField


class SearchQueryGenerator(dspy.Signature):
    """Generate optimized search queries to verify a factual statement.

    Given a statement to verify, generate two different search queries that
    will help find relevant evidence. Use different angles or phrasings to
    maximize the chance of finding comprehensive information.
    """

    statement: str = InputField(desc="The statement to research and verify")

    query1: str = OutputField(desc="Primary search query to find evidence")
    query2: str = OutputField(desc="Alternative search query from different angle")


class EvidenceSummarizer(dspy.Signature):
    """Summarize scraped web content into relevant evidence for fact-checking.

    Extract and condense the most relevant information from web sources that
    relates to verifying the given statement. Focus on factual claims and
    ignore irrelevant content.
    """

    statement: str = InputField(desc="The original statement being verified")
    raw_content: str = InputField(desc="Raw scraped content from web sources")

    summary: str = OutputField(desc="Concise summary of relevant evidence extracted from the content")
