"""Signatures for the fact checker."""

from src.factchecker.signatures.judge import Judge
from src.factchecker.signatures.search_query_generator import SearchQueryGenerator
from src.factchecker.signatures.evidence_aware_judge import EvidenceAwareJudge
from src.factchecker.signatures.source_deep_dive import SourceDeepDive

__all__ = ["Judge", "SearchQueryGenerator", "EvidenceAwareJudge", "SourceDeepDive"]
