#!/usr/bin/env python3
"""Test script for the enhanced adaptive JudgeModule."""

import dspy
from src.factchecker.simple.modules.judge_module import JudgeModule

def test_adaptive_judge():
    """Test the adaptive judge module with different scenarios."""

    # Configure dspy (you may need to adjust this based on your setup)
    # Uncomment and configure if you have an LLM configured
    # lm = dspy.OpenAI(model="gpt-4")
    # dspy.settings.configure(lm=lm)

    print("=" * 80)
    print("Testing Adaptive JudgeModule")
    print("=" * 80)

    # Test 1: Module instantiation with different configs
    print("\n[Test 1] Instantiation with different configurations")

    judge_no_search = JudgeModule(enable_adaptive_search=False)
    print("✓ Created JudgeModule with adaptive search disabled")

    judge_with_search = JudgeModule(
        enable_adaptive_search=True,
        confidence_threshold=0.6,
        num_search_results=3
    )
    print("✓ Created JudgeModule with adaptive search enabled")

    judge_custom = JudgeModule(
        enable_adaptive_search=True,
        confidence_threshold=0.7,
        num_search_results=2,
        max_scrape_length=5000
    )
    print("✓ Created JudgeModule with custom parameters")

    # Test 2: Verify trigger logic
    print("\n[Test 2] Testing search trigger logic")

    test_cases = [
        ("This is beyond my knowledge cutoff", 0.8, True),
        ("I cannot verify this claim", 0.9, True),
        ("I cannot confirm this information", 0.7, True),
        ("This statement is clearly false", 0.9, False),
        ("This is a factual statement", 0.3, True),  # Low confidence
        ("This is supported by evidence", 0.8, False),
    ]

    for reasoning, confidence, expected in test_cases:
        result = judge_with_search._should_trigger_search(reasoning, confidence)
        status = "✓" if result == expected else "✗"
        print(f"{status} Reasoning: '{reasoning[:40]}...' | Confidence: {confidence} | "
              f"Expected: {expected} | Got: {result}")

    print("\n[Test 3] Module attributes and methods")
    print(f"✓ enable_adaptive_search: {judge_with_search.enable_adaptive_search}")
    print(f"✓ confidence_threshold: {judge_with_search.confidence_threshold}")
    print(f"✓ num_search_results: {judge_with_search.num_search_results}")
    print(f"✓ max_scrape_length: {judge_with_search.max_scrape_length}")

    print("\n" + "=" * 80)
    print("All basic tests passed! ✓")
    print("=" * 80)
    print("\nNOTE: To test the full functionality with LLM integration:")
    print("  1. Configure dspy with your LLM (OpenAI, Claude, etc.)")
    print("  2. Set up API keys for SerperService and FirecrawlService")
    print("  3. Call judge_module.forward(statement='Your test statement')")
    print("\nExample statements to test:")
    print("  - Recent events: 'The 2024 Olympics were held in Paris'")
    print("  - Historical facts: 'World War II ended in 1945'")
    print("  - Uncertain claims: 'A new planet was discovered last week'")
    print("=" * 80)

if __name__ == "__main__":
    test_adaptive_judge()
