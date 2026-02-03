"""Test script for enhanced JudgeModule with web search capability.

This script demonstrates how the enhanced JudgeModule handles recent events
and facts beyond the LLM's training data by automatically performing web
searches when knowledge limitations are detected.
"""

import dspy
from src.context_.context import openai_key
from src.factchecker.simple.modules.judge_module import JudgeModule


def configure_dspy(model: str = "openai/gpt-4o-mini"):
    """Configure DSPy with the specified model."""
    lm = dspy.LM(model, api_key=openai_key)
    dspy.configure(lm=lm)
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)


def test_judge_module():
    """Test the enhanced JudgeModule with various statements."""

    print("=" * 80)
    print("Testing Enhanced JudgeModule with Web Search Capability")
    print("=" * 80)

    # Configure DSPy
    configure_dspy("openai/gpt-5-mini")

    # Test cases: mix of old facts, recent events, and false claims
    test_statements = [
        # Should NOT trigger search - historical fact
        "The Great Wall of China was built over several centuries starting in the 7th century BC.",

        # Should trigger search - 2025 event (example)
        "OpenAI released GPT-5 in early 2025 with significantly improved reasoning capabilities.",

        # Should trigger search - recent event
        "Donald Trump won the 2024 U.S. presidential election.",

        # Should NOT trigger search - clear false statement
        "The Earth is flat and orbits around the Moon.",

        # Should trigger search - uncertain recent event
        "SpaceX launched its first crewed mission to Mars in 2025.",
    ]

    # Test with web search enabled
    print("\n" + "=" * 80)
    print("TEST 1: JudgeModule WITH Web Search (enable_web_search=True)")
    print("=" * 80)

    judge_with_search = JudgeModule(enable_web_search=True)

    for i, statement in enumerate(test_statements, 1):
        print(f"\n{'-'*80}")
        print(f"Statement {i}: {statement}")
        print(f"{'-'*80}")

        result = judge_with_search(statement=statement)

        print(f"Verdict: {result.overall_verdict}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Web Search Performed: {result.web_search_performed}")
        print(f"Reasoning: {result.reasoning}")

    # Test with web search disabled (original behavior)
    print("\n\n" + "=" * 80)
    print("TEST 2: JudgeModule WITHOUT Web Search (enable_web_search=False)")
    print("=" * 80)
    print("Testing one statement to show original behavior without search:")
    print(f"{'-'*80}")

    judge_without_search = JudgeModule(enable_web_search=False)

    # Test a 2025 statement that should show uncertainty without search
    test_statement = "Donald Trump won the 2024 U.S. presidential election."
    print(f"Statement: {test_statement}")
    print(f"{'-'*80}")

    result = judge_without_search(statement=test_statement)

    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Web Search Performed: {result.web_search_performed}")
    print(f"Reasoning: {result.reasoning}")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_judge_module()
