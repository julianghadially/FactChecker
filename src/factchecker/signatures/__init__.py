"""DSPy signatures for fact-checking operations."""

from .claim_extractor import ClaimExtractor
from .fire_judge import FireJudge
from .page_selector import PageSelector
from .evidence_summarizer import EvidenceSummarizer
from .aggregator import Aggregator
from .search_query_generator import SearchQueryGenerator

__all__ = [
    "ClaimExtractor",
    "FireJudge",
    "PageSelector",
    "EvidenceSummarizer",
    "Aggregator",
    "SearchQueryGenerator",
]
