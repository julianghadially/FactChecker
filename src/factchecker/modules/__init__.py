"""Modules for the fact checker."""

from src.factchecker.modules.judge_module import JudgeModule
from src.factchecker.modules.claim_type_analyzer_module import ClaimTypeAnalyzerModule
from src.factchecker.modules.search_query_generator_module import SearchQueryGeneratorModule
from src.factchecker.modules.evidence_retriever_module import EvidenceRetrieverModule

__all__ = [
    "JudgeModule",
    "ClaimTypeAnalyzerModule",
    "SearchQueryGeneratorModule",
    "EvidenceRetrieverModule",
]
