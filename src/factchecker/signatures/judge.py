"""Simple judge signature for direct statement evaluation without research."""

from dspy import Signature, InputField, OutputField
from typing import Literal


class Judge(Signature):
    """Evaluate a statement's factual correctness without external research.

    Assess whether the statement is factually accurate based on your knowledge.
    Output one of three verdicts:
    - SUPPORTED: The statement is factually correct
    - CONTAINS_REFUTED_CLAIMS: The statement contains false information
    - CONTAINS_UNSUPPORTED_CLAIMS: Cannot determine - insufficient knowledge
    """

    statement: str = InputField(desc="The statement to evaluate for factual correctness")
    topic: str = InputField(
        default="",
        desc="Optional context: The topic or domain this statement relates to (e.g., 'Alaska Air', 'Politics')"
    )
    date: str = InputField(
        default="",
        desc="Optional context: The date when this statement was generated or refers to (YYYYMMDD format)"
    )
    source_urls: str = InputField(
        default="",
        desc="Optional context: Comma-separated URLs that provide relevant context for this statement"
    )

    reasoning: str = OutputField(desc="Explanation of why this verdict was chosen")
    verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"] = OutputField(
        desc="The factual correctness verdict for the statement"
    )
    confidence: float = OutputField(desc="Confidence score between 0.0 and 1.0")
