"""Context enricher signature for extracting relevant evidence from URLs."""

from dspy import Signature, InputField, OutputField


class ContextEnricher(Signature):
    """Extract and summarize evidence from web content that relates to a statement.

    Given a statement and scraped web page content, identify and extract relevant
    information that provides context or evidence for evaluating the statement.
    Focus on factual information with proper source attribution.
    """

    statement: str = InputField(desc="The statement being evaluated")
    page_content: str = InputField(desc="Markdown content scraped from the web page")
    source_url: str = InputField(desc="URL of the source page for attribution")

    relevant_evidence: str = OutputField(
        desc="Extracted facts and context relevant to the statement, with source attribution"
    )
