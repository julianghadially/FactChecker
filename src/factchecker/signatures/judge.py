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

    Optional contextual metadata may be provided to enhance evaluation:
    - topic: Use to narrow the domain and apply specialized knowledge in that area
    - date_generated: Use to assess temporal relevance (e.g., was this true at that time?)
    - url: Use as a hint about the claim's origin and potential context

    When metadata is provided, leverage these contextual hints to make more informed judgments.
    """

    statement: str = InputField(desc="The statement to evaluate for factual correctness")
    topic: Optional[str] = InputField(default=None, desc="Optional topic/domain context to narrow evaluation scope")
    url: Optional[str] = InputField(default=None, desc="Optional URL hint about the claim's origin")
    date_generated: Optional[str] = InputField(default=None, desc="Optional date context for temporal relevance assessment")

    reasoning: str = OutputField(desc="Explanation of why this verdict was chosen")
    verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"] = OutputField(
        desc="The factual correctness verdict for the statement"
    )
    confidence: float = OutputField(desc="Confidence score between 0.0 and 1.0")
