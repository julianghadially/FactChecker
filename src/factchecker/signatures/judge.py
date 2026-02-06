"""Simple judge signature for direct statement evaluation without research."""

from dspy import Signature, InputField, OutputField
from typing import Literal


class Judge(Signature):
    """Evaluate a statement's factual correctness without external research.

    Assess whether the statement is factually accurate based on your knowledge.
    Context information (topic, url, date_generated) may be provided to help
    understand the statement's domain and timeframe. These fields may be empty.

    Output one of three verdicts:
    - SUPPORTED: The statement is factually correct
    - CONTAINS_REFUTED_CLAIMS: The statement contains false information
    - CONTAINS_UNSUPPORTED_CLAIMS: Cannot determine - insufficient knowledge
    """

    statement: str = InputField(desc="The statement to evaluate for factual correctness")
    topic: str = InputField(desc="The topic/domain of the statement (may be empty)")
    url: str = InputField(desc="Reference URL for the statement (may be empty)")
    date_generated: str = InputField(desc="Date when the statement was created (may be empty)")

    reasoning: str = OutputField(desc="Explanation of why this verdict was chosen")
    verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"] = OutputField(
        desc="The factual correctness verdict for the statement"
    )
    confidence: float = OutputField(desc="Confidence score between 0.0 and 1.0")
