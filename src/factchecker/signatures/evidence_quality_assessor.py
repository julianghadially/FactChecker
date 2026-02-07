"""Evidence quality assessor signature for evaluating evidence sufficiency."""

from dspy import Signature, InputField, OutputField


class EvidenceQualityAssessor(Signature):
    """Assess whether gathered evidence is sufficient to fact-check a statement.

    Analyze the statement and the evidence collected so far to determine:
    1. What specific claims in the statement are covered by the evidence
    2. What specific claims are missing or inadequately addressed
    3. Whether the evidence is sufficient to make a reliable judgment
    4. What targeted follow-up queries would fill the gaps

    This assessment helps implement adaptive search: if initial broad queries return
    off-topic results or failed scrapes, generate 1-2 targeted follow-up queries to
    retrieve specific information needed (e.g., corporate agreements, technical specs,
    specific dates or figures).

    Quality assessment should:
    - Identify which claims have supporting/refuting evidence
    - Pinpoint specific information gaps (e.g., "missing: X Corp partnership details")
    - Recognize when evidence is off-topic or tangentially related
    - Be conservative: mark insufficient when critical claims lack direct evidence

    Follow-up queries should:
    - Be highly specific and targeted to fill identified gaps
    - Include precise keywords, company names, technical terms, or dates
    - Focus on the most critical missing information
    - Be different from initial queries (more specific/targeted)
    """

    statement: str = InputField(desc="The statement being fact-checked")
    evidence: str = InputField(desc="Markdown content from web sources gathered so far")

    quality_assessment: str = OutputField(
        desc="Detailed explanation of what claims are covered, what's missing, and why evidence is sufficient/insufficient"
    )
    is_sufficient: bool = OutputField(
        desc="Whether the evidence is sufficient to reliably fact-check the statement (true) or more evidence is needed (false)"
    )
    followup_queries: list[str] = OutputField(
        desc="1-2 highly targeted search queries to fill specific evidence gaps (empty list if is_sufficient=true)"
    )
