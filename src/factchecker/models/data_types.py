"""Shared data types for the fact-checker system."""

from dataclasses import dataclass
from typing import Literal


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
