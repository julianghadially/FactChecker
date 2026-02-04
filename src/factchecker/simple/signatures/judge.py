"""Simple judge signature for direct statement evaluation without research."""

from dspy import Signature, InputField, OutputField
from typing import Literal


class Judge(Signature):
    """Evaluate a statement's factual correctness using a two-stage reasoning process.

    STAGE 1: Check if external evidence is provided in the evidence field.
    - If evidence is provided: Base your verdict primarily on that evidence, treating it as authoritative source material.
    - If evidence is empty: Proceed to Stage 2.

    STAGE 2: Fall back to internal knowledge only when evidence field is empty.
    - Assess whether the statement is factually accurate based on your knowledge.

    Output one of three verdicts:
    - SUPPORTED: The statement is factually correct
    - CONTAINS_REFUTED_CLAIMS: The statement contains false information
    - CONTAINS_UNSUPPORTED_CLAIMS: Cannot determine - insufficient knowledge
    """

    statement: str = InputField(desc="The statement to evaluate for factual correctness")
    evidence: str = InputField(desc="External evidence from web sources (leave empty if unavailable). If provided, this evidence should be treated as authoritative and take precedence over internal knowledge in your reasoning.", default="")

    reasoning: str = OutputField(desc="Explanation of why this verdict was chosen")
    verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"] = OutputField(
        desc="The factual correctness verdict for the statement"
    )
    confidence: float = OutputField(desc="Confidence score between 0.0 and 1.0")
