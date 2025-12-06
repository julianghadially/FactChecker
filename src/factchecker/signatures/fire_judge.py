"""Fire Judge signature for iterative fact verification."""

from dspy import Signature, InputField, OutputField
from typing import Literal, Optional


class FireJudge(Signature):
    """FIRE (Fact-checking with Iterative Research and Evaluation) Judge.

    Evaluate a single factual claim given accumulated evidence from web research.
    Either produce a final verdict if sufficient evidence exists, or generate
    a search query to gather more information.

    Decision logic:
    - If evidence clearly supports the claim -> verdict="supported"
    - If evidence clearly contradicts the claim -> verdict="refuted"
    - If evidence is insufficient but a useful search is possible -> next_search=<query>
    - If no more useful searches and evidence is inconclusive -> verdict="not_supported"
    """

    claim: str = InputField(desc="A single factual claim to verify")
    evidence: str = InputField(desc="Evidence gathered from web research, may be empty initially")
    search_history: list[str] = InputField(desc="Previous search queries already executed")

    reasoning: str = OutputField(desc="Step-by-step reasoning about the claim and evidence")
    verdict: Optional[Literal["supported", "not_supported", "refuted"]] = OutputField(
        desc="Final judgment if enough evidence exists, otherwise None"
    )
    next_search: Optional[str] = OutputField(
        desc="Search query if more evidence needed, otherwise None. Must differ from search_history."
    )