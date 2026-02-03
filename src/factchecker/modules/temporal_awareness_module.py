"""Temporal Awareness module for detecting time-sensitive claims."""

import dspy
from datetime import datetime
from typing import Optional
from src.factchecker.models.data_types import TemporalContext


class TemporalAnalysis(dspy.Signature):
    """Analyze a statement for temporal signals and determine if it's beyond knowledge cutoff.

    Extract dates, years, and temporal phrases (like "recently", "this year", "last month", etc.).
    Determine if the statement refers to events or data beyond June 2024, which would require
    web search to verify.

    Knowledge cutoff date: June 2024
    Current date: {current_date}
    """

    statement: str = dspy.InputField(desc="The statement to analyze for temporal signals")
    temporal_entities: list[str] = dspy.OutputField(
        desc="List of detected temporal references (dates, years, phrases like 'recently', 'this year', etc.)"
    )
    is_beyond_cutoff: bool = dspy.OutputField(
        desc="True if the statement refers to events/data after June 2024, or if temporal uncertainty is high for recent events"
    )
    reasoning: str = dspy.OutputField(
        desc="Explanation of why this statement is or isn't beyond the knowledge cutoff"
    )
    suggested_year_filter: Optional[str] = dspy.OutputField(
        desc="If beyond cutoff, suggest a year filter for search queries (e.g., '2025', '2024')"
    )
    is_news_relevant: bool = dspy.OutputField(
        desc="True if this appears to be about recent news/events that would benefit from news search"
    )


class TemporalAwarenessModule(dspy.Module):
    """Module that detects temporal signals and determines if web search is needed.

    This module preprocesses statements to identify time-sensitive claims that are
    beyond the LLM's knowledge cutoff (June 2024). When such claims are detected,
    it generates context that instructs downstream modules to perform temporal-aware
    web searches.

    Attributes:
        knowledge_cutoff: The date of the LLM's knowledge cutoff (default: June 2024).
        analyzer: DSPy chain for temporal analysis.
    """

    def __init__(self, knowledge_cutoff_date: str = "2024-06-01"):
        """Initialize the temporal awareness module.

        Args:
            knowledge_cutoff_date: Knowledge cutoff date in YYYY-MM-DD format.
        """
        super().__init__()
        self.knowledge_cutoff = datetime.strptime(knowledge_cutoff_date, "%Y-%m-%d")
        self.current_date = datetime.now()

        # Update the signature docstring with current date
        TemporalAnalysis.__doc__ = TemporalAnalysis.__doc__.format(
            current_date=self.current_date.strftime("%B %d, %Y")
        )

        self.analyzer = dspy.ChainOfThought(TemporalAnalysis)

    def forward(self, statement: str) -> TemporalContext:
        """Analyze a statement for temporal signals and generate search context.

        Args:
            statement: The statement to analyze.

        Returns:
            TemporalContext with analysis results and suggested search strategies.
        """
        # Run temporal analysis
        result = self.analyzer(statement=statement)

        # Generate search modifiers based on analysis
        search_modifiers = []
        context_parts = []

        if result.is_beyond_cutoff:
            context_parts.append(
                "⚠️ TEMPORAL AWARENESS: This claim contains references to events or data "
                f"beyond the knowledge cutoff (June 2024)."
            )

            # Add year filter suggestion
            if result.suggested_year_filter:
                search_modifiers.append(f"Add year filter: {result.suggested_year_filter}")
                context_parts.append(
                    f"When searching, prioritize results from {result.suggested_year_filter}."
                )

            # Add news search suggestion for recent events
            if result.is_news_relevant:
                search_modifiers.append("Use news search for recent events")
                context_parts.append(
                    "Consider using SerperService.search_news() to find recent news articles "
                    "about this topic with temporal filters (recency='d', 'w', or 'm')."
                )

            # Add temporal entities to context
            if result.temporal_entities:
                context_parts.append(
                    f"Detected temporal references: {', '.join(result.temporal_entities)}"
                )

            context_parts.append(
                "🌐 ACTION REQUIRED: You MUST perform web searches to verify this claim. "
                "Do not rely solely on pre-existing knowledge."
            )
            context_parts.append(f"Reasoning: {result.reasoning}")

        context_message = "\n".join(context_parts) if context_parts else ""

        return TemporalContext(
            has_temporal_signals=len(result.temporal_entities) > 0,
            is_beyond_cutoff=result.is_beyond_cutoff,
            temporal_entities=result.temporal_entities,
            suggested_search_modifiers=search_modifiers,
            context_message=context_message
        )
