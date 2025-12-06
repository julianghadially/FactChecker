"""Baseline model that evaluates claims using only LLM knowledge."""

import dspy
from typing import Literal


class BaselineFactCheck(dspy.Signature):
    """Determine if a claim is factually correct based on your knowledge.

    Evaluate the claim using only your training data knowledge, without
    access to web search or external information sources.
    """

    claim: str = dspy.InputField(desc="A factual claim to evaluate")

    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning about the claim")
    verdict: Literal["SUPPORTED", "NOT_ENOUGH_INFO", "REFUTED"] = dspy.OutputField(
        desc="Your judgment: SUPPORTED, NOT_ENOUGH_INFO, or REFUTED"
    )


class BaselineModel(dspy.Module):
    """Simple baseline that relies solely on LLM knowledge without web search.

    This model serves as a comparison baseline to demonstrate the value
    of grounded fact-checking with web search.
    """

    def __init__(self):
        """Initialize the baseline model."""
        super().__init__()
        self.predictor = dspy.ChainOfThought(BaselineFactCheck)

    def forward(self, claim: str) -> dict:
        """Evaluate a claim using only LLM knowledge.

        Args:
            claim: The claim to evaluate.

        Returns:
            Dict with 'claim', 'verdict', and 'reasoning' keys.
        """
        result = self.predictor(claim=claim)
        return {
            "claim": claim,
            "verdict": result.verdict,
            "reasoning": result.reasoning
        }
