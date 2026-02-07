"""Evidence quality assessor module for evaluating evidence sufficiency."""

import dspy
from src.factchecker.signatures.evidence_quality_assessor import EvidenceQualityAssessor


class EvidenceQualityAssessorModule(dspy.Module):
    """Module that assesses whether gathered evidence is sufficient for fact-checking.

    Takes a statement and collected evidence as input and uses an LLM to:
    1. Analyze which claims are covered vs. missing in the evidence
    2. Determine if evidence is sufficient to make a reliable judgment
    3. Generate targeted follow-up queries to fill specific gaps if needed

    This enables adaptive search: instead of giving up when initial broad queries
    return off-topic results or failed scrapes, the system can identify what specific
    information is missing and generate targeted queries to retrieve it.

    This is inserted between evidence retrieval (stage 2) and judgment (stage 3)
    in the fact-checking pipeline.
    """

    def __init__(self):
        """Initialize the evidence quality assessor module."""
        super().__init__()
        self.assessor = dspy.ChainOfThought(EvidenceQualityAssessor)

    def forward(self, statement: str, evidence: str) -> dspy.Prediction:
        """Assess evidence quality and generate follow-up queries if needed.

        Args:
            statement: The statement being fact-checked.
            evidence: The evidence gathered so far (markdown content with sources).

        Returns:
            dspy.Prediction with:
                - quality_assessment: Explanation of what's covered/missing
                - is_sufficient: Boolean indicating if more evidence is needed
                - followup_queries: List of 1-2 targeted queries (empty if sufficient)
        """
        result = self.assessor(statement=statement, evidence=evidence)

        return dspy.Prediction(
            quality_assessment=result.quality_assessment,
            is_sufficient=result.is_sufficient,
            followup_queries=result.followup_queries if hasattr(result, 'followup_queries') else [],
        )
