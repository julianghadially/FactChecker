"""Aggregator signature for combining claim verdicts into overall statement verdict."""

from dspy import Signature, InputField, OutputField
from typing import Literal


class Aggregator(Signature):
    """Aggregate individual claim verdicts into an overall statement verdict.

    Apply the following priority logic:
    1. If ANY claim is refuted -> CONTAINS_REFUTED_CLAIMS (highest priority)
    2. If ANY claim is not_supported -> CONTAINS_UNSUPPORTED_CLAIMS
    3. If ALL claims are supported -> SUPPORTED
    """

    original_statement: str = InputField(desc="The original statement being evaluated")
    claim_verdicts: list[dict] = InputField(
        desc="List of dicts with 'claim', 'verdict', and 'evidence_summary' keys"
    )

    reasoning: str = OutputField(desc="Explanation of the aggregation logic applied")
    overall_verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"] = OutputField(
        desc="The final verdict for the entire statement"
    )
    confidence: float = OutputField(desc="Confidence score between 0.0 and 1.0")
