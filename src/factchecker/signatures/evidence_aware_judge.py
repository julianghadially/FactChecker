"""Evidence-aware judge signature for evaluating statements with web evidence."""

from dspy import Signature, InputField, OutputField
from typing import Literal


class EvidenceAwareJudge(Signature):
    """Evaluate a statement's factual correctness using gathered web evidence.

    Assess whether the statement is factually accurate by analyzing the provided
    evidence gathered from authoritative web sources. Compare the claims in the
    statement against the evidence to determine if they are supported, refuted,
    or cannot be verified.

    Output one of three verdicts:
    - SUPPORTED: The statement is factually correct according to the evidence
    - CONTAINS_REFUTED_CLAIMS: The statement contains information contradicted by evidence
    - CONTAINS_UNSUPPORTED_CLAIMS: Evidence is insufficient to verify the claims

    Your reasoning should:
    - Cite specific evidence that supports or contradicts the statement
    - Reference source URLs when making claims about what evidence shows
    - Explain any discrepancies or contradictions found
    - Acknowledge when evidence is insufficient or ambiguous
    """

    statement: str = InputField(desc="The statement to evaluate for factual correctness")
    evidence: str = InputField(desc="Markdown content from web sources with source attribution")

    reasoning: str = OutputField(desc="Explanation citing specific evidence and sources that led to this verdict")
    verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"] = OutputField(
        desc="The factual correctness verdict based on evidence"
    )
    confidence: float = OutputField(desc="Confidence score between 0.0 and 1.0")
