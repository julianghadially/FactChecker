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
    evidence_context: str = InputField(
        default="",
        desc="Additional evidence context from web sources to help evaluate the statement"
    )

    evidence_analysis: str = OutputField(
        desc="First, extract and analyze ALL relevant facts from the evidence_context that support or contradict the statement. List specific quotes, dates, numbers, and sources. If evidence_context is empty or insufficient, explicitly state what information is missing."
    )
    reasoning: str = OutputField(
        desc="Based on the evidence_analysis above, explain why this verdict was chosen and how the evidence supports your conclusion."
    )
    verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"] = OutputField(
        desc="The factual correctness verdict for the statement"
    )
    confidence: float = OutputField(desc="Confidence score between 0.0 and 1.0")
