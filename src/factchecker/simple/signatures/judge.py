"""Simple judge signature for direct statement evaluation without research."""

from dspy import Signature, InputField, OutputField
from typing import Literal


class Judge(Signature):
    """Evaluate a statement's factual correctness using parametric knowledge and web evidence.

    Assess whether the statement is factually accurate based on your knowledge.
    If web evidence is provided in the statement context, incorporate it into your analysis.

    Output one of three verdicts:
    - SUPPORTED: The statement is factually correct
    - CONTAINS_REFUTED_CLAIMS: The statement contains false information
    - CONTAINS_UNSUPPORTED_CLAIMS: Cannot determine - insufficient knowledge

    Explicitly indicate whether external verification (web search) is needed by setting
    the needs_external_verification field to True when you lack sufficient information
    due to knowledge cutoff, temporal limitations, or uncertainty.
    """

    statement: str = InputField(desc="The statement to evaluate for factual correctness")

    reasoning: str = OutputField(desc="Explanation of why this verdict was chosen")
    verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"] = OutputField(
        desc="The factual correctness verdict for the statement"
    )
    confidence: float = OutputField(desc="Confidence score between 0.0 and 1.0")
    needs_external_verification: bool = OutputField(
        desc="True if web search is needed to verify this statement due to knowledge cutoff, recency, or uncertainty; False if parametric knowledge is sufficient"
    )
