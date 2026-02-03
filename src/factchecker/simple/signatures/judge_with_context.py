"""Judge signature with web search context for enhanced fact verification."""

from dspy import Signature, InputField, OutputField
from typing import Literal


class JudgeWithContext(Signature):
    """Evaluate a statement's factual correctness with web search context.

    This signature extends the basic Judge by including recent information from
    web search results, allowing verification of claims beyond the LLM's training
    data cutoff.

    Assess whether the statement is factually accurate based on both your knowledge
    AND the provided search results. The search results contain recent information
    that may be beyond your training data.

    Output one of three verdicts:
    - SUPPORTED: The statement is factually correct based on the search results
    - CONTAINS_REFUTED_CLAIMS: The statement contains false information per the search results
    - CONTAINS_UNSUPPORTED_CLAIMS: Cannot determine - insufficient or conflicting information
    """

    statement: str = InputField(desc="The statement to evaluate for factual correctness")
    search_results: str = InputField(
        desc="Recent web search results providing context for verification. "
        "Each result includes: title, URL, and snippet with relevant information."
    )
    initial_reasoning: str = InputField(
        desc="The initial reasoning that indicated knowledge limitations or uncertainty"
    )

    reasoning: str = OutputField(
        desc="Explanation of why this verdict was chosen, citing specific search results"
    )
    verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"] = OutputField(
        desc="The factual correctness verdict for the statement"
    )
    confidence: float = OutputField(desc="Confidence score between 0.0 and 1.0")
