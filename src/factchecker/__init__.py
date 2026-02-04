"""FactChecker - DSPy-based multi-hop fact verification system."""

from .modules.judge_module import JudgeModule
from .models.data_types import JudgmentResult

__all__ = [
    "JudgeModule",
    "JudgmentResult",
]
