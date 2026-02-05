"""Signatures for the simple fact checker."""

from src.factchecker.signatures.judge import Judge
from src.factchecker.signatures.research import SearchQueryGenerator, EvidenceSummarizer

__all__ = ["Judge", "SearchQueryGenerator", "EvidenceSummarizer"]
