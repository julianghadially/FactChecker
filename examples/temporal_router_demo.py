"""Demo script showcasing the TemporalRouterModule capabilities.

This script demonstrates how the TemporalRouterModule intelligently routes
fact-checking requests based on temporal references and provided URLs.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from src.factchecker.modules.temporal_router_module import TemporalRouterModule


def configure_dspy():
    """Configure DSPy with appropriate model."""
    # Note: You'll need to set OPENAI_API_KEY environment variable
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)


def demo_temporal_routing():
    """Demonstrate temporal routing with various test cases."""

    print("=" * 80)
    print("TEMPORAL ROUTER MODULE DEMO")
    print("=" * 80)

    # Initialize the router
    router = TemporalRouterModule()

    # Test Case 1: Historical statement (should use JudgeModule)
    print("\n" + "=" * 80)
    print("TEST CASE 1: Historical Statement (Pre-Cutoff)")
    print("=" * 80)
    statement1 = "The Apollo 11 mission landed on the moon on July 20, 1969."
    print(f"\nStatement: {statement1}")

    result1 = router(statement=statement1)
    print(f"\nRoute Decision: {result1.route_decision}")
    print(f"Route Reason: {result1.route_reason}")
    print(f"Verdict: {result1.overall_verdict}")
    print(f"Reasoning: {result1.reasoning}")

    # Test Case 2: Recent statement (should use FactCheckerPipeline)
    print("\n" + "=" * 80)
    print("TEST CASE 2: Recent Statement (Post-Cutoff)")
    print("=" * 80)
    statement2 = "In January 2025, the global tech industry saw major layoffs."
    print(f"\nStatement: {statement2}")

    result2 = router(statement=statement2)
    print(f"\nRoute Decision: {result2.route_decision}")
    print(f"Route Reason: {result2.route_reason}")
    print(f"Verdict: {result2.overall_verdict}")

    # Test Case 3: Statement with temporal keywords (should use FactCheckerPipeline)
    print("\n" + "=" * 80)
    print("TEST CASE 3: Statement with Temporal Keywords")
    print("=" * 80)
    statement3 = "The latest climate report shows record temperatures this year."
    print(f"\nStatement: {statement3}")

    result3 = router(statement=statement3)
    print(f"\nRoute Decision: {result3.route_decision}")
    print(f"Route Reason: {result3.route_reason}")
    print(f"Verdict: {result3.overall_verdict}")

    # Test Case 4: Statement with URLs (should use FactCheckerPipeline)
    print("\n" + "=" * 80)
    print("TEST CASE 4: Statement with URLs")
    print("=" * 80)
    statement4 = "According to https://example.com/news, the economy is growing."
    print(f"\nStatement: {statement4}")

    result4 = router(statement=statement4)
    print(f"\nRoute Decision: {result4.route_decision}")
    print(f"Route Reason: {result4.route_reason}")
    print(f"Verdict: {result4.overall_verdict}")

    # Test Case 5: Explicit URL list (should use FactCheckerPipeline with priority)
    print("\n" + "=" * 80)
    print("TEST CASE 5: Statement with Explicit Priority URLs")
    print("=" * 80)
    statement5 = "The company announced record profits in Q4."
    priority_urls = [
        "https://www.example.com/earnings-report",
        "https://www.example.com/financial-news"
    ]
    print(f"\nStatement: {statement5}")
    print(f"Priority URLs: {priority_urls}")

    result5 = router(statement=statement5, urls=priority_urls)
    print(f"\nRoute Decision: {result5.route_decision}")
    print(f"Route Reason: {result5.route_reason}")
    print(f"Verdict: {result5.overall_verdict}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


def test_date_extraction():
    """Test the date extraction functionality."""
    print("\n" + "=" * 80)
    print("DATE EXTRACTION TEST")
    print("=" * 80)

    router = TemporalRouterModule()

    test_statements = [
        "On January 15, 2025, the event occurred.",
        "The year 2024 was significant.",
        "This happened on 2025-03-20.",
        "In March 2026, things will change.",
        "The Apollo mission was in 1969.",
    ]

    for stmt in test_statements:
        dates = router._extract_dates(stmt)
        print(f"\nStatement: {stmt}")
        print(f"Extracted dates: {[d.strftime('%Y-%m-%d') for d in dates]}")
        should_use_web, reason = router._should_use_web_research(stmt, [], dates)
        print(f"Use web research: {should_use_web} ({reason})")


if __name__ == "__main__":
    # Configure DSPy
    try:
        configure_dspy()
    except Exception as e:
        print(f"Error configuring DSPy: {e}")
        print("Make sure OPENAI_API_KEY environment variable is set.")
        sys.exit(1)

    # Run date extraction test (no API calls)
    test_date_extraction()

    # Uncomment below to run full demo (requires API calls)
    # demo_temporal_routing()
