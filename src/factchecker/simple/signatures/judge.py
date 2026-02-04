"""Simple judge signature for direct statement evaluation without research."""

from dspy import Signature, InputField, OutputField
from typing import Literal


class Judge(Signature):
    """Evaluate a statement's factual correctness using provided evidence sources.

    CRITICAL: You MUST use the provided evidence context to evaluate the statement.
    Do NOT default to knowledge cutoff limitations when evidence is available.
    Base your verdict on the evidence sources provided, not solely on your training data.

    Output one of three verdicts:
    - SUPPORTED: The statement is factually correct based on evidence
    - CONTAINS_REFUTED_CLAIMS: The statement contains false information per evidence
    - CONTAINS_UNSUPPORTED_CLAIMS: Cannot determine - evidence insufficient or unavailable
    """

    statement: str = InputField(desc="The statement to evaluate for factual correctness")
    evidence_context: str = InputField(
        desc="Evidence context from web sources that MUST be used to evaluate the statement. When provided, base your verdict on this evidence rather than knowledge cutoff limitations."
    )

    reasoning: str = OutputField(desc="Explanation of why this verdict was chosen")
    evidence_usage: str = OutputField(
        desc="Explicit description of how the evidence context was used (or why it wasn't used) in reaching the verdict. You must engage with the provided evidence and explain which specific facts informed your decision."
    )
    verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"] = OutputField(
        desc="The factual correctness verdict for the statement"
    )
    confidence: float = OutputField(desc="Confidence score between 0.0 and 1.0")
