"""Baseline model that evaluates claims using only LLM knowledge."""

import dspy
from typing import Literal


class BaselineFactCheck(dspy.Signature):
    """Determine if a claim is factually correct based on your knowledge.

    Evaluate the claim using only your training data knowledge, without
    access to web search or external information sources.
    Context information (topic, url, date_generated) may be provided to help
    understand the claim's domain and timeframe. These fields may be empty.
    """

    claim: str = dspy.InputField(desc="A factual claim to evaluate")
    topic: str = dspy.InputField(desc="The topic/domain of the claim (may be empty)")
    url: str = dspy.InputField(desc="Reference URL for the claim (may be empty)")
    date_generated: str = dspy.InputField(desc="Date when the claim was created (may be empty)")

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

    def forward(
        self,
        statement: str,
        topic: str = "",
        url: str = "",
        date_generated: str = ""
    ) -> dict:
        """Evaluate a claim using only LLM knowledge.

        Args:
            statement: The claim to evaluate.
            topic: Optional topic/domain context (default: "").
            url: Optional reference URL (default: "").
            date_generated: Optional creation date (default: "").

        Returns:
            Dict with 'claim', 'verdict', and 'reasoning' keys.
        """
        # technically could convert statement to claims for a full baseline, But the data sets are one claim at a time so it's OK.
        claim = statement
        result = self.predictor(
            claim=claim,
            topic=topic,
            url=url,
            date_generated=date_generated
        )
        return {
            "claim": claim,
            "verdict": result.verdict,
            "reasoning": result.reasoning
        }
