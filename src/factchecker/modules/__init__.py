"""DSPy modules for fact-checking operations."""

from .claim_extractor_module import ClaimExtractorModule
from .research_agent_module import ResearchAgentModule
from .fire_judge_module import FireJudgeModule
from .aggregator_module import AggregatorModule
from .fact_checker_pipeline import FactCheckerPipeline

__all__ = [
    "ClaimExtractorModule",
    "ResearchAgentModule",
    "FireJudgeModule",
    "AggregatorModule",
    "FactCheckerPipeline",
]
