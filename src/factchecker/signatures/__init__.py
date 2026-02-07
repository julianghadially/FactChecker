"""Signatures for the fact checker."""

from src.factchecker.signatures.judge import Judge
from src.factchecker.signatures.claim_type_analyzer import ClaimTypeAnalyzer
from src.factchecker.signatures.search_query_generator import SearchQueryGenerator
from src.factchecker.signatures.evidence_aware_judge import EvidenceAwareJudge

__all__ = ["Judge", "ClaimTypeAnalyzer", "SearchQueryGenerator", "EvidenceAwareJudge"]
