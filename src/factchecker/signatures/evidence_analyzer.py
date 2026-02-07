"""Evidence analyzer signature for structured analysis of evidence."""

from dspy import Signature, InputField, OutputField


class EvidenceAnalyzer(Signature):
    """Perform structured analysis of evidence to extract facts, detect contradictions, and compute values.

    This signature provides systematic reasoning capabilities for handling claims that require:
    - Numerical computation (e.g., summing donations, comparing values)
    - Logical contradiction detection (e.g., finding a single fact that refutes a claim)
    - Fact extraction (e.g., identifying key dates, names, numerical values)

    The analysis should:
    - Extract all relevant numerical values, dates, and names from evidence
    - Identify any direct contradictions between evidence and statement
    - Perform necessary arithmetic (sums, comparisons, calculations)
    - Synthesize findings into a structured summary for the judge

    This enables the judge to make more accurate verdicts on claims like:
    - "Total donations were less than $10M" (requires summing individual donations)
    - "Event occurred in 2020" (requires checking if any source contradicts this)
    - "Company has 5 offices" (requires counting and verifying)
    """

    statement: str = InputField(desc="The statement being fact-checked")
    evidence: str = InputField(desc="Markdown content from web sources with source attribution")

    extracted_facts: list[str] = OutputField(
        desc="List of key facts extracted from evidence including: numerical values with units, dates, names, locations, and other concrete claims. Each fact should cite its source."
    )
    logical_contradictions: list[str] = OutputField(
        desc="List of direct contradictions between the statement and evidence. Each contradiction should cite specific evidence that refutes a claim in the statement. Empty list if no contradictions found."
    )
    numerical_computations: str = OutputField(
        desc="Any arithmetic or numerical analysis needed to verify the statement (e.g., 'Sum of donations: $5M + $7M + $15M = $27M, which exceeds the claimed $10M limit'). Include the specific calculation and result. Use 'N/A' if no computations needed."
    )
    synthesis: str = OutputField(
        desc="Structured summary of the analysis for the judge, highlighting: (1) key facts that support or refute the statement, (2) any contradictions found, (3) results of numerical computations, and (4) overall assessment of evidence quality for this specific claim"
    )
