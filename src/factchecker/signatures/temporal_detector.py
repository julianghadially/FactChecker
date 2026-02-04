"""Temporal claim detection signature for identifying time-sensitive statements."""

from dspy import Signature, InputField, OutputField


class TemporalDetector(Signature):
    """Detect if a statement contains temporal claims requiring recent knowledge.

    Analyze statements for references to:
    - Dates/years >= 2024 (after June 2024 knowledge cutoff)
    - Future events or predictions
    - Recent events described with temporal indicators (e.g., "recently", "this year")
    - Current status claims that could change over time

    Examples requiring web research:
    - "In 2024, the GDP growth rate was..."
    - "The current president of..."
    - "Recent studies show that..."
    - "This year's Nobel Prize winner is..."

    Examples NOT requiring web research:
    - "World War II ended in 1945"
    - "Water boils at 100°C at sea level"
    - "Shakespeare wrote Hamlet"
    """

    statement: str = InputField(desc="The statement to analyze for temporal claims")

    reasoning: str = OutputField(desc="Explanation of temporal indicators found or why the statement is not time-sensitive")
    requires_recent_knowledge: bool = OutputField(
        desc="True if the statement requires knowledge after June 2024 or refers to current/future events"
    )
