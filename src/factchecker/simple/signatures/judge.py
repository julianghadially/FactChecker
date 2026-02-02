"""Simple judge signature for direct statement evaluation without research."""

from dspy import Signature, InputField, OutputField
from typing import Literal


class Judge(Signature):
    """Evaluate a statement's factual correctness without external research.

    Assess whether the statement is factually accurate AND sufficiently precise based on your knowledge.
    A statement must be COMPLETE, UNAMBIGUOUS, and provide necessary context to be SUPPORTED.
    If a statement contains accurate data but lacks critical qualifiers (time period, population subset, units, conditions), it should be treated as imprecise and marked REFUTED or UNSUPPORTED.

    Output one of three verdicts:
    - SUPPORTED: The statement is factually correct, complete, and unambiguous
    - CONTAINS_REFUTED_CLAIMS: The statement is false, misleading, or critically imprecise
    - CONTAINS_UNSUPPORTED_CLAIMS: Cannot determine - insufficient knowledge
    """

    statement: str = InputField(desc="The statement to evaluate for factual correctness")

    reasoning: str = OutputField(desc="Explanation of why this verdict was chosen")
    verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"] = OutputField(
        desc="The factual correctness verdict for the statement"
    )
    confidence: float = OutputField(desc="Confidence score between 0.0 and 1.0")
