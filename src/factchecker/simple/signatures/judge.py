"""Simple judge signature for direct statement evaluation without research."""

from dspy import Signature, InputField, OutputField
from typing import Literal


class Judge(Signature):
    """Evaluate a statement's factual correctness with mandatory evidence-based reasoning.

    CRITICAL: When evidence_context is provided, it is the PRIMARY source of truth and MUST be
    analyzed first before considering any internal knowledge. You must engage directly with the
    provided evidence through a multi-step reasoning process.

    Assess whether the statement is factually accurate based on the provided evidence and your knowledge.
    Output one of three verdicts:
    - SUPPORTED: The statement is factually correct
    - CONTAINS_REFUTED_CLAIMS: The statement contains false information
    - CONTAINS_UNSUPPORTED_CLAIMS: Cannot determine - insufficient knowledge
    """

    statement: str = InputField(desc="The statement to evaluate for factual correctness")
    evidence_context: str = InputField(
        default="",
        desc="Additional evidence context from web sources to help evaluate the statement. When provided, this is the PRIMARY source of truth and must be analyzed before considering internal knowledge."
    )

    evidence_analysis: str = OutputField(
        desc="REQUIRED: Extract and quote specific facts from evidence_context that are relevant to the statement. You MUST cite actual text from the provided evidence using quotes, or explicitly state 'No evidence provided' if evidence_context is empty. Do not skip this step - actually engage with the evidence text."
    )
    evidence_comparison: str = OutputField(
        desc="REQUIRED: Compare the statement's specific claims against the facts cited in evidence_analysis. For each claim in the statement, explicitly state whether the cited evidence supports it, refutes it, or is silent on it. Be specific about what matches or contradicts."
    )
    reasoning: str = OutputField(
        desc="Synthesize findings from evidence_analysis and evidence_comparison to reach a final verdict. Explain how the evidence-based analysis leads to your conclusion. If evidence was provided, your reasoning must directly reference the evidence analysis and comparison above."
    )
    verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"] = OutputField(
        desc="The factual correctness verdict for the statement"
    )
    confidence: float = OutputField(desc="Confidence score between 0.0 and 1.0")
