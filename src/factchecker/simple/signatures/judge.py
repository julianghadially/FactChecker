"""Simple judge signature for direct statement evaluation without research."""

from dspy import Signature, InputField, OutputField
from typing import Literal, Optional


class Judge(Signature):
    """Evaluate a statement's factual correctness without external research.

    Assess whether the statement is factually accurate based on your knowledge.
    Output one of three verdicts:
    - SUPPORTED: The statement is factually correct
    - CONTAINS_REFUTED_CLAIMS: The statement contains false information
    - CONTAINS_UNSUPPORTED_CLAIMS: Cannot determine - insufficient knowledge

    IMPORTANT: If the statement references events, data, or timeframes after the
    knowledge_cutoff_date, you should return CONTAINS_UNSUPPORTED_CLAIMS since you
    lack current information to verify claims about recent events.
    """

    statement: str = InputField(desc="The statement to evaluate for factual correctness")
    statement_date: Optional[str] = InputField(
        desc="Date the statement was made/generated (format: YYYYMMDD). Used to assess temporal context."
    )
    knowledge_cutoff_date: str = InputField(
        desc="The LLM's knowledge cutoff date (format: YYYYMMDD). Claims about events after this date should return CONTAINS_UNSUPPORTED_CLAIMS."
    )

    reasoning: str = OutputField(desc="Explanation of why this verdict was chosen")
    verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"] = OutputField(
        desc="The factual correctness verdict for the statement"
    )
    confidence: float = OutputField(desc="Confidence score between 0.0 and 1.0")
