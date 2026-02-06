"""FactChecker - DSPy-based multi-hop fact verification system."""

from .modules.judge_module import JudgeModule
from .modules.research_module import ResearchModule
from .modules.factchecker_pipeline import FactCheckerPipeline
from .models.data_types import JudgmentResult

__all__ = [
    "JudgeModule",
    "ResearchModule",
    "FactCheckerPipeline",
    "JudgmentResult",
]
