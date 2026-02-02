"""Simple judge signature for direct statement evaluation without research."""

from dspy import Signature, InputField, OutputField
from typing import Literal


class Judge(Signature):
    """Evaluate a statement's factual correctness based on your internal knowledge.

    Use your knowledge confidently. If the statement aligns with what you know (even if you're not 100% certain of every minor detail like exact dates), mark it SUPPORTED. Only use UNSUPPORTED when you genuinely lack knowledge about the topic or entities mentioned.

    Output one of three verdicts:
    - SUPPORTED: The statement aligns with your knowledge and appears factually correct
    - CONTAINS_REFUTED_CLAIMS: The statement contradicts what you know - it contains false information
    - CONTAINS_UNSUPPORTED_CLAIMS: You lack sufficient knowledge about this topic/entity to make a determination
    """

    statement: str = InputField(desc="The statement to evaluate for factual correctness")

    reasoning: str = OutputField(desc="Explanation of why this verdict was chosen")
    verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"] = OutputField(
        desc="The factual correctness verdict for the statement"
    )
    confidence: float = OutputField(desc="Confidence score between 0.0 and 1.0")
