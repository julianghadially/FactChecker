#!/usr/bin/env python3
"""
Example usage of the enhanced adaptive JudgeModule.

This script demonstrates how to use the JudgeModule with adaptive web search
for fact-checking various types of statements.

Before running:
1. Configure dspy with your LLM (e.g., OpenAI, Anthropic Claude)
2. Set environment variables: SERPER_API_KEY, FIRECRAWL_API_KEY
"""

import dspy
from src.factchecker.simple.modules.judge_module import JudgeModule


def example_basic_usage():
    """Basic usage example."""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)

    # Create judge with default adaptive search settings
    judge = JudgeModule(enable_adaptive_search=True)

    # Test statements
    statements = [
        "The Earth revolves around the Sun",
        "The 2024 Summer Olympics were held in Paris",
        "The James Webb Space Telescope launched in December 2021",
    ]

    for stmt in statements:
        print(f"\nStatement: {stmt}")
        print("-" * 80)

        # Note: Uncomment below when you have dspy configured with an LLM
        # result = judge.forward(stmt)
        # print(f"Verdict: {result.overall_verdict}")
        # print(f"Confidence: {result.confidence:.2f}")
        # print(f"Web search triggered: {result.web_search_triggered}")
        # print(f"Reasoning: {result.reasoning[:200]}...")

        print("(Uncomment code and configure dspy to see actual results)")


def example_configuration_options():
    """Example showing different configuration options."""
    print("\n" + "=" * 80)
    print("Example 2: Configuration Options")
    print("=" * 80)

    # Conservative: Only search when very uncertain
    conservative_judge = JudgeModule(
        enable_adaptive_search=True,
        confidence_threshold=0.4,  # Low threshold = fewer searches
        num_search_results=2
    )
    print("✓ Created conservative judge (fewer searches)")

    # Aggressive: Search more frequently for higher accuracy
    aggressive_judge = JudgeModule(
        enable_adaptive_search=True,
        confidence_threshold=0.8,  # High threshold = more searches
        num_search_results=5,
        max_scrape_length=10000
    )
    print("✓ Created aggressive judge (more searches, more evidence)")

    # Disabled: No web search (fastest, original behavior)
    simple_judge = JudgeModule(enable_adaptive_search=False)
    print("✓ Created simple judge (no web search)")


def example_analyzing_results():
    """Example showing how to analyze results with metadata."""
    print("\n" + "=" * 80)
    print("Example 3: Analyzing Results with Metadata")
    print("=" * 80)

    judge = JudgeModule(enable_adaptive_search=True)

    # Mixed batch of statements
    test_cases = [
        ("Historical fact", "World War II ended in 1945"),
        ("Recent event", "The 2024 US presidential election was held in November"),
        ("Scientific fact", "Water boils at 100 degrees Celsius at sea level"),
        ("Uncertain claim", "A new exoplanet was discovered last month"),
    ]

    print("\nProcessing batch of statements...\n")

    results_summary = []
    for category, stmt in test_cases:
        print(f"Category: {category}")
        print(f"Statement: {stmt}")

        # Uncomment when dspy is configured
        # result = judge.forward(stmt)
        # results_summary.append({
        #     "category": category,
        #     "statement": stmt,
        #     "verdict": result.overall_verdict,
        #     "confidence": result.confidence,
        #     "searched": result.web_search_triggered,
        #     "initial_conf": result.initial_confidence if result.web_search_triggered else None
        # })
        #
        # print(f"  → Verdict: {result.overall_verdict}")
        # print(f"  → Confidence: {result.confidence:.2f}")
        # print(f"  → Web search: {'Yes' if result.web_search_triggered else 'No'}")
        # if result.web_search_triggered:
        #     print(f"  → Initial confidence: {result.initial_confidence:.2f}")
        #     print(f"  → Confidence improvement: {result.confidence - result.initial_confidence:.2f}")

        print("  (Configure dspy to see results)")
        print()

    # Uncomment to see analytics
    # print("\n" + "=" * 80)
    # print("Batch Analytics")
    # print("=" * 80)
    # search_rate = sum(r["searched"] for r in results_summary) / len(results_summary)
    # avg_confidence = sum(r["confidence"] for r in results_summary) / len(results_summary)
    # print(f"Search trigger rate: {search_rate:.1%}")
    # print(f"Average confidence: {avg_confidence:.2f}")


def example_error_handling():
    """Example showing error handling."""
    print("\n" + "=" * 80)
    print("Example 4: Error Handling")
    print("=" * 80)

    # Judge will work even without API keys (won't trigger search though)
    judge = JudgeModule(enable_adaptive_search=True)

    # If API keys are missing, search will fail gracefully
    # The error will be included in the evidence field
    print("✓ Module handles missing API keys gracefully")
    print("✓ Errors are captured in result.evidence field")
    print("✓ Module falls back to LLM-only judgment")


def example_custom_workflow():
    """Example showing custom workflow with the judge."""
    print("\n" + "=" * 80)
    print("Example 5: Custom Workflow")
    print("=" * 80)

    judge = JudgeModule(enable_adaptive_search=True)

    # Custom workflow: Check statement, and if search was triggered,
    # log it for review
    def check_statement_with_logging(statement: str):
        """Check statement and log when web search is used."""
        # result = judge.forward(statement)
        #
        # if result.web_search_triggered:
        #     # Log searches for review/audit
        #     print(f"[SEARCH LOG] Statement: {statement}")
        #     print(f"[SEARCH LOG] Initial confidence: {result.initial_confidence:.2f}")
        #     print(f"[SEARCH LOG] Final confidence: {result.confidence:.2f}")
        #     print(f"[SEARCH LOG] Verdict: {result.overall_verdict}")
        #     # Could write to file, database, etc.
        #
        # return result

        print("(Configure dspy to see custom workflow in action)")

    check_statement_with_logging("Example statement about recent events")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("ADAPTIVE JUDGE MODULE - USAGE EXAMPLES")
    print("=" * 80)
    print("\nNote: These examples show the structure and usage patterns.")
    print("To see actual results, you need to:")
    print("  1. Configure dspy with an LLM: dspy.settings.configure(lm=your_lm)")
    print("  2. Set API keys: SERPER_API_KEY, FIRECRAWL_API_KEY")
    print("  3. Uncomment the result = judge.forward(...) lines")
    print()

    example_basic_usage()
    example_configuration_options()
    example_analyzing_results()
    example_error_handling()
    example_custom_workflow()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review ADAPTIVE_JUDGE_ENHANCEMENT.md for full documentation")
    print("  2. Configure your environment with LLM and API keys")
    print("  3. Uncomment the code in this file to see live results")
    print("  4. Experiment with different configuration options")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
