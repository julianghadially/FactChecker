"""Evidence analysis module for structured reasoning about evidence."""

import dspy
from src.factchecker.signatures.evidence_analyzer import EvidenceAnalyzer


class EvidenceAnalysisModule(dspy.Module):
    """Module that performs structured analysis of evidence before judgment.

    This module adds systematic reasoning capabilities to handle claims requiring:
    - Numerical computation (summing values, comparing numbers)
    - Logical contradiction detection (finding refuting facts)
    - Fact extraction (identifying key data points)

    Takes a statement and evidence as input and uses an LLM with Chain-of-Thought
    reasoning to:
    1. Extract all relevant facts with numerical values, dates, names
    2. Identify direct contradictions between statement and evidence
    3. Perform necessary arithmetic or numerical analysis
    4. Synthesize findings into a structured summary

    This is inserted between evidence quality assessment (stage 2.5) and judgment
    (stage 4) in the fact-checking pipeline. The analysis output is passed to the
    judge to enable more accurate verdicts on numerically complex or logically
    subtle claims.

    Examples of claims this helps with:
    - "Company received less than $10M total" → sums individual donations
    - "Event had no accidents" → detects single contradictory fact
    - "Product released in Q2 2023" → extracts and verifies date
    """

    def __init__(self):
        """Initialize the evidence analysis module."""
        super().__init__()
        self.analyzer = dspy.ChainOfThought(EvidenceAnalyzer)

    def forward(self, statement: str, evidence: str) -> dspy.Prediction:
        """Analyze evidence to extract facts, detect contradictions, and compute values.

        Args:
            statement: The statement being fact-checked.
            evidence: The evidence gathered (markdown content with sources).

        Returns:
            dspy.Prediction with:
                - extracted_facts: List of key facts from evidence with sources
                - logical_contradictions: List of contradictions found (empty if none)
                - numerical_computations: Arithmetic analysis if needed (or 'N/A')
                - synthesis: Structured summary for the judge
        """
        result = self.analyzer(statement=statement, evidence=evidence)

        return dspy.Prediction(
            extracted_facts=result.extracted_facts if hasattr(result, 'extracted_facts') else [],
            logical_contradictions=result.logical_contradictions if hasattr(result, 'logical_contradictions') else [],
            numerical_computations=result.numerical_computations if hasattr(result, 'numerical_computations') else "N/A",
            synthesis=result.synthesis if hasattr(result, 'synthesis') else "",
        )
