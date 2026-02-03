"""Example demonstrating the enhanced JudgeModule with web search capability.

This example shows how the JudgeModule now handles statements that require
recent information by automatically falling back to web search when knowledge
cutoff limitations are detected.
"""

import dspy
from src.factchecker.simple.modules.judge_module import JudgeModule

# Configure DSPy with your LLM
# dspy.configure(lm=your_lm_instance)

def example_basic_judgment():
    """Example: Basic judgment without web search needed."""
    print("=== Example 1: Basic Judgment (No Web Search Needed) ===")
    judge = JudgeModule(use_web_search=True)

    statement = "The Earth orbits around the Sun."
    result = judge(statement=statement)

    print(f"Statement: {statement}")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence}")
    print(f"Web Evidence Used: {result.web_evidence_used}")
    print(f"Reasoning: {result.reasoning}")
    print()


def example_with_web_search():
    """Example: Judgment requiring web search for recent events."""
    print("=== Example 2: Recent Event (Web Search Expected) ===")
    judge = JudgeModule(use_web_search=True)

    statement = "SpaceX launched Starship Flight 6 in November 2024."
    result = judge(statement=statement)

    print(f"Statement: {statement}")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence}")
    print(f"Web Evidence Used: {result.web_evidence_used}")
    print(f"Reasoning: {result.reasoning}")
    print()


def example_web_search_disabled():
    """Example: Web search disabled, will use only parametric knowledge."""
    print("=== Example 3: Web Search Disabled ===")
    judge = JudgeModule(use_web_search=False)

    statement = "The 2024 US Presidential election was held in November."
    result = judge(statement=statement)

    print(f"Statement: {statement}")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence}")
    print(f"Web Evidence Used: {result.web_evidence_used}")
    print(f"Reasoning: {result.reasoning}")
    print()


if __name__ == "__main__":
    print("Enhanced JudgeModule Examples")
    print("=" * 60)
    print()

    # Note: You need to configure DSPy with your LLM before running these examples
    # Uncomment the examples below after configuration:

    # example_basic_judgment()
    # example_with_web_search()
    # example_web_search_disabled()

    print("To run these examples:")
    print("1. Configure DSPy with your LLM: dspy.configure(lm=your_lm)")
    print("2. Ensure SERPER_API_KEY and FIRECRAWL_API_KEY are set")
    print("3. Uncomment the example function calls above")
