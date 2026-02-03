"""Fire Judge signature for iterative fact verification."""

from dspy import Signature, InputField, OutputField
from typing import Literal, Optional

class FireJudge(Signature):
    """FIRE (Fact-checking with Iterative Research and Evaluation) Judge.

    Evaluate a single factual claim given accumulated evidence from web research.
    Either produce a final verdict if sufficient evidence exists, or generate
    a search query to gather more information.

    CRITICAL INSTRUCTIONS FOR HANDLING KNOWLEDGE CUTOFF:
    1. Your knowledge cutoff is April 2024. Claims about events, data, or developments
       after this date are BEYOND your internal knowledge.
    2. When a claim involves dates/events after April 2024, you MUST generate a web
       search query (next_search) rather than returning verdict="not_supported" based
       on lack of internal knowledge alone.
    3. For time-specific claims, ALWAYS include relevant dates/timeframes in your
       search queries. Examples:
       - Claim about "BlackRock Bitcoin ETF returns in 2025" -> Search: "BlackRock Bitcoin ETF returns 2025"
       - Claim about "Skechers World Champions Cup winner" -> Search: "Skechers World Champions Cup 2025 winner"
       - Claim about "November 2025 event" -> Search: "November 2025 [event details]"
    4. Use the current_date field to understand what timeframe to search for and to
       determine if a claim falls within or beyond your knowledge cutoff.
    5. Only return verdict="not_supported" when you have exhausted useful search
       queries AND the available evidence remains inconclusive.

    Decision logic:
    - If evidence clearly supports the claim -> verdict="supported"
    - If evidence clearly contradicts the claim -> verdict="refuted"
    - If claim involves dates/events after April 2024 and no evidence gathered yet -> next_search=<query with dates>
    - If evidence is insufficient but a useful search is possible -> next_search=<query>
    - If no more useful searches and evidence is inconclusive -> verdict="not_supported"
    """

    claim: str = InputField(desc="A single factual claim to verify")
    current_date: str = InputField(desc="Current date to provide context for time-specific claims and knowledge cutoff awareness")
    evidence: str = InputField(desc="Evidence gathered from web research, may be empty initially")
    search_history: list[str] = InputField(desc="Previous search queries already executed")
    reasoning: str = OutputField(desc="Step-by-step reasoning about the claim and evidence")
    verdict: Optional[Literal["supported", "not_supported", "refuted"]] = OutputField(
        desc="Final judgment if enough evidence exists, otherwise None"
    )
    next_search: Optional[str] = OutputField(
        desc="Search query if more evidence needed, otherwise None. Must differ from search_history."
    )