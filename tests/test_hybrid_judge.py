"""Test script for HybridJudgeModule to verify routing logic."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from src.context_.context import openai_key
from src.factchecker.modules.hybrid_judge_module import HybridJudgeModule


def test_hybrid_routing():
    """Test that HybridJudgeModule routes claims correctly."""

    # Configure DSPy
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key=openai_key))

    # Initialize hybrid module
    hybrid = HybridJudgeModule(max_judge_iterations=2, max_page_visits=2)

    print("=" * 80)
    print("HYBRID JUDGE MODULE TEST")
    print("=" * 80)

    # Test case 1: Recent temporal claim (should use web search)
    print("\n" + "=" * 80)
    print("TEST 1: Recent temporal claim")
    print("=" * 80)
    statement1 = "In December 2025, OpenAI announced GPT-5 with 10 trillion parameters."
    print(f"Statement: {statement1}")
    result1 = hybrid(statement=statement1)
    print(f"\nRouting Decision: {result1.routing_decision}")
    print(f"Routing Reasoning: {result1.routing_reasoning}")
    print(f"Overall Verdict: {result1.overall_verdict}")
    print(f"Confidence: {result1.confidence}")
    print(f"Reasoning: {result1.reasoning}")
    assert result1.routing_decision == "web_search", "Should route to web search for recent claims"

    # Test case 2: General knowledge (should use simple judge)
    print("\n" + "=" * 80)
    print("TEST 2: General knowledge claim")
    print("=" * 80)
    statement2 = "Paris is the capital of France."
    print(f"Statement: {statement2}")
    result2 = hybrid(statement=statement2)
    print(f"\nRouting Decision: {result2.routing_decision}")
    print(f"Routing Reasoning: {result2.routing_reasoning}")
    print(f"Overall Verdict: {result2.overall_verdict}")
    print(f"Confidence: {result2.confidence}")
    print(f"Reasoning: {result2.reasoning}")
    assert result2.routing_decision == "simple_judge", "Should route to simple judge for general knowledge"

    # Test case 3: Company-specific recent claim (should use web search)
    print("\n" + "=" * 80)
    print("TEST 3: Company-specific claim")
    print("=" * 80)
    statement3 = "Apple's board approved a $150 billion stock buyback in Q4 2025."
    print(f"Statement: {statement3}")
    result3 = hybrid(statement=statement3)
    print(f"\nRouting Decision: {result3.routing_decision}")
    print(f"Routing Reasoning: {result3.routing_reasoning}")
    print(f"Overall Verdict: {result3.overall_verdict}")
    print(f"Confidence: {result3.confidence}")
    print(f"Reasoning: {result3.reasoning}")
    assert result3.routing_decision == "web_search", "Should route to web search for company claims"

    # Test case 4: Historical fact (should use simple judge)
    print("\n" + "=" * 80)
    print("TEST 4: Historical fact")
    print("=" * 80)
    statement4 = "World War II ended in 1945."
    print(f"Statement: {statement4}")
    result4 = hybrid(statement=statement4)
    print(f"\nRouting Decision: {result4.routing_decision}")
    print(f"Routing Reasoning: {result4.routing_reasoning}")
    print(f"Overall Verdict: {result4.overall_verdict}")
    print(f"Confidence: {result4.confidence}")
    print(f"Reasoning: {result4.reasoning}")
    assert result4.routing_decision == "simple_judge", "Should route to simple judge for historical facts"

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    test_hybrid_routing()
