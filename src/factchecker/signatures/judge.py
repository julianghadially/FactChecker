"""Simple judge signature for direct statement evaluation without research."""

from dspy import Signature, InputField, OutputField
from typing import Literal


class Judge(Signature):
    """Evaluate a statement's factual correctness using provided evidence from web sources.

    Assess whether the statement is factually accurate based on the provided evidence
    and your knowledge. When evidence is available, prioritize it over training data.
    Output one of three verdicts:
    - SUPPORTED: The statement is factually correct
    - CONTAINS_REFUTED_CLAIMS: The statement contains false information
    - CONTAINS_UNSUPPORTED_CLAIMS: Cannot determine - insufficient knowledge
    """

    statement: str = InputField(desc="The statement to evaluate for factual correctness")
    evidence: str = InputField(desc="Relevant evidence from web sources (empty if no research performed)")

    reasoning: str = OutputField(desc="Explanation of why this verdict was chosen, citing specific evidence when available")
    verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"] = OutputField(
        desc="The factual correctness verdict for the statement"
    )
    confidence: float = OutputField(desc="Confidence score between 0.0 and 1.0")
