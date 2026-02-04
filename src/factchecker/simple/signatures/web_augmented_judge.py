"""Web-augmented judge signature for statement evaluation with evidence."""

from dspy import Signature, InputField, OutputField
from typing import Literal


class WebAugmentedJudge(Signature):
    """Evaluate a statement's factual correctness with supporting web evidence.

    Re-assess a statement using evidence gathered from web search and scraping.
    This is used when the initial LLM-only judgment was uncertain or lacked
    information due to knowledge cutoff.

    Output one of three verdicts:
    - SUPPORTED: The statement is factually correct based on the evidence
    - CONTAINS_REFUTED_CLAIMS: The statement contains false information
    - CONTAINS_UNSUPPORTED_CLAIMS: Evidence is insufficient or inconclusive
    """

    statement: str = InputField(desc="The statement to evaluate for factual correctness")
    evidence: str = InputField(desc="Relevant evidence from web search and scraped pages")

    reasoning: str = OutputField(desc="Explanation of why this verdict was chosen based on the evidence")
    verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"] = OutputField(
        desc="The factual correctness verdict for the statement"
    )
    confidence: float = OutputField(desc="Confidence score between 0.0 and 1.0")
