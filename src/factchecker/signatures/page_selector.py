"""Page selector signature for intelligent page selection from search results."""

from dspy import Signature, InputField, OutputField
from typing import Optional


class PageSelector(Signature):
    """Select the most promising page to visit from search results for evidence gathering.

    Given a claim being fact-checked and search results, intelligently select which
    page to visit next based on relevance, authoritativeness, and potential to
    provide supporting or refuting evidence.
    """

    claim: str = InputField(desc="The factual claim being verified")
    search_results: list[dict] = InputField(
        desc="Search results with 'title', 'link', 'snippet' fields"
    )
    visited_urls: list[str] = InputField(desc="URLs already visited in this research session")
    current_evidence: str = InputField(desc="Evidence already gathered from previous pages")

    reasoning: str = OutputField(desc="Explanation of why this page is most relevant to the claim")
    selected_url: Optional[str] = OutputField(
        desc="URL to visit next, or None if existing evidence is sufficient or no useful pages remain unvisited"
    )
