"""Simple judge signature for direct statement evaluation without research."""

from dspy import Signature, InputField, OutputField
from typing import Literal


class Judge(Signature):
    """Evaluate a statement's factual correctness based on world knowledge.

    Carefully assess whether the statement is factually accurate. Use these rules:

    - SUPPORTED: Use when the statement is factually correct or broadly accepted as true,
      even if it uses hedging language like "typically", "generally", "often", "tend to",
      or "usually". A statement qualifies as SUPPORTED if the core claim is true in the
      majority of cases or reflects established consensus. Do NOT require a precise
      comparison target or exhaustive caveats to mark a statement as SUPPORTED.

    - CONTAINS_REFUTED_CLAIMS: Use when the statement contains information that is
      demonstrably false, contradicts established facts, or is clearly incorrect based on
      well-known knowledge.

    - CONTAINS_UNSUPPORTED_CLAIMS: Use ONLY as a last resort when you genuinely cannot
      determine whether the statement is true or false — e.g., highly niche claims,
      very recent events beyond your knowledge, or claims where expert consensus is
      completely divided. Do NOT use this verdict merely because a statement is vague,
      uses comparative language without naming a comparison target, or lacks full
      precision. Prefer a definitive verdict (SUPPORTED or CONTAINS_REFUTED_CLAIMS)
      whenever reasonably possible.
    """

    statement: str = InputField(desc="The statement to evaluate for factual correctness")

    reasoning: str = OutputField(desc="Step-by-step analysis of the statement's factual accuracy, citing relevant world knowledge")
    verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"] = OutputField(
        desc="The factual correctness verdict: SUPPORTED if the claim is broadly true, CONTAINS_REFUTED_CLAIMS if it contains false information, CONTAINS_UNSUPPORTED_CLAIMS ONLY if you truly cannot determine truth from world knowledge"
    )
    confidence: float = OutputField(desc="Confidence score between 0.0 and 1.0")
