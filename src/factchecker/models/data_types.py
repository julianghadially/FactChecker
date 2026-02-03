"""Shared data types for the fact-checker system."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class TemporalContext:
    """Context about temporal aspects of a statement.

    Attributes:
        has_temporal_signals: Whether temporal references were detected.
        is_beyond_cutoff: Whether dates/events are beyond knowledge cutoff (June 2024).
        temporal_entities: List of detected dates, years, or temporal phrases.
        suggested_search_modifiers: Suggested query modifications for web search.
        context_message: Human-readable context for the judge module.
    """
    has_temporal_signals: bool
    is_beyond_cutoff: bool
    temporal_entities: list[str]
    suggested_search_modifiers: list[str]
    context_message: str


@dataclass
class JudgmentResult:
    """Result from evaluating a single claim."""

    claim: str
    verdict: Literal["supported", "not_supported", "refuted"]
    evidence_summary: str
    search_queries: list[str]
    iterations: int


@dataclass
class AggregationResult:
    """Result from aggregating multiple claim verdicts."""

    verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"]
    confidence: float
    reasoning: str
    claim_details: list[dict]


@dataclass
class FactCheckResult:
    """Complete result from fact-checking a statement."""

    statement: str
    claims: list[str]
    claim_results: list[JudgmentResult]
    overall_verdict: Literal["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"]
    confidence: float
    reasoning: str
