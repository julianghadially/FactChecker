"""FactChecker - DSPy-based multi-hop fact verification system."""

from .modules.judge_module import JudgeModule
from .modules.research_module import ResearchModule
from .models.data_types import JudgmentResult, ResearchResult

__all__ = [
    "JudgeModule",
    "ResearchModule",
    "JudgmentResult",
    "ResearchResult",
]
