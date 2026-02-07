"""Modules for the fact checker."""

from src.factchecker.modules.judge_module import JudgeModule
from src.factchecker.modules.search_query_generator_module import SearchQueryGeneratorModule
from src.factchecker.modules.evidence_retriever_module import EvidenceRetrieverModule
from src.factchecker.modules.source_deep_dive_module import SourceDeepDiveModule

__all__ = ["JudgeModule", "SearchQueryGeneratorModule", "EvidenceRetrieverModule", "SourceDeepDiveModule"]
