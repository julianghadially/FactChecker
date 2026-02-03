"""Evidence summarizer signature for extracting relevant facts from web pages."""

from dspy import Signature, InputField, OutputField


class EvidenceSummarizer(Signature):
    """Extract and summarize evidence relevant to verifying a specific claim.

    Given a claim and scraped web page content, identify and extract facts
    that either support or refute the claim. Focus on factual information
    with proper source attribution.
    """

    claim: str = InputField(desc="The factual claim being verified")
    page_content: str = InputField(desc="Markdown content scraped from the web page")
    source_url: str = InputField(desc="URL of the source page for attribution")

    relevant_evidence: str = OutputField(
        desc="Extracted facts relevant to the claim, with source attribution"
    )
    evidence_stance: str = OutputField(
        desc="Whether this evidence 'supports', 'refutes', or is 'neutral' toward the claim"
    )
    temporal_context: str = OutputField(
        desc="Any dates, years, time references, or temporal information extracted from the evidence (e.g., 'Data from 2020', 'As of January 2024', 'Updated monthly'). This helps identify if data is current or outdated."
    )
