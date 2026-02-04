"""Simple judge signature for direct statement evaluation without research."""

from dspy import Signature, InputField, OutputField
from typing import Literal


class Judge(Signature):
    """Evaluate a statement's factual correctness by analyzing provided evidence. When evidence_context is provided, base your verdict primarily on that evidence. Only rely on your own knowledge when no evidence is available."""

    statement: str = InputField(desc="The statement to evaluate for factual correctness")
    evidence_context: str = InputField(
        default="",
        desc="Additional evidence context from web sources to help evaluate the statement"
    )

    evidence_analysis: str = OutputField(
        desc="First, analyze what the provided evidence says about the statement. If no evidence was provided, state that you will rely on your knowledge."
    )
    reasoning: str = OutputField(desc="Explanation of why this verdict was chosen")
    verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"] = OutputField(
        desc="The factual correctness verdict for the statement"
    )
    confidence: float = OutputField(desc="Confidence score between 0.0 and 1.0")
