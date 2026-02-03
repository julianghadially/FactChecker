"""Example demonstrating the TemporalAwarenessModule functionality.

This script shows how the TemporalAwarenessModule detects temporal signals
in statements and provides context for web-based fact-checking.
"""

import dspy
from src.factchecker.modules.temporal_awareness_module import TemporalAwarenessModule
from src.context_.context import anthropic_key

# Configure DSPy
dspy.settings.configure(lm=dspy.LM("anthropic/claude-3-5-sonnet-20241022", api_key=anthropic_key))


def demonstrate_temporal_awareness():
    """Demonstrate the TemporalAwarenessModule with various statements."""

    # Initialize the module
    temporal_module = TemporalAwarenessModule()

    # Test statements with different temporal characteristics
    test_statements = [
        # Statement about 2025 (beyond cutoff)
        "The 2025 US presidential inauguration occurred on January 20, 2025.",

        # Statement about recent 2024 events (near/beyond cutoff)
        "The 2024 Summer Olympics in Paris concluded in August 2024.",

        # Statement with relative temporal phrases
        "Apple recently announced a new iPhone model this year.",

        # Statement about historical facts (before cutoff)
        "The Apollo 11 mission landed on the moon on July 20, 1969.",

        # Statement with no temporal signals
        "The Earth orbits around the Sun.",
    ]

    print("=" * 80)
    print("TEMPORAL AWARENESS MODULE DEMONSTRATION")
    print("=" * 80)
    print()

    for i, statement in enumerate(test_statements, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i}: {statement}")
        print(f"{'=' * 80}\n")

        # Run temporal analysis
        context = temporal_module(statement=statement)

        # Display results
        print(f"📊 Has Temporal Signals: {context.has_temporal_signals}")
        print(f"⚠️  Beyond Cutoff: {context.is_beyond_cutoff}")
        print(f"📅 Temporal Entities: {context.temporal_entities}")
        print(f"🔍 Search Modifiers: {context.suggested_search_modifiers}")

        if context.context_message:
            print(f"\n📝 Context Message:")
            print(f"{'-' * 80}")
            print(context.context_message)
            print(f"{'-' * 80}")

        print()


def demonstrate_integration_with_pipeline():
    """Show how temporal context integrates with the full pipeline."""

    print("\n" + "=" * 80)
    print("INTEGRATION WITH FACT CHECKER PIPELINE")
    print("=" * 80)
    print()

    from src.factchecker.modules.fact_checker_pipeline import FactCheckerPipeline

    # Initialize pipeline (which now includes TemporalAwarenessModule)
    pipeline = FactCheckerPipeline(max_judge_iterations=2, max_page_visits=2)

    # Test with a statement about 2025
    statement = "Donald Trump was inaugurated as the 47th President of the United States on January 20, 2025."

    print(f"Statement to fact-check:\n{statement}\n")
    print("Running fact-checking pipeline with temporal awareness...")
    print()

    # Run the pipeline (temporal awareness is now automatically integrated)
    result = pipeline(statement=statement)

    print(f"\n{'=' * 80}")
    print("PIPELINE RESULTS")
    print(f"{'=' * 80}")
    print(f"Overall Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"\nReasoning:\n{result.reasoning}")
    print()

    # Show claim-level details
    print("Claim-Level Results:")
    for i, claim_result in enumerate(result.claim_results, 1):
        print(f"\n  Claim {i}: {claim_result.claim}")
        print(f"  Verdict: {claim_result.verdict}")
        print(f"  Search Queries: {claim_result.search_queries}")
        print(f"  Iterations: {claim_result.iterations}")


if __name__ == "__main__":
    # Demonstrate the temporal awareness module
    demonstrate_temporal_awareness()

    # Optionally demonstrate full pipeline integration
    # Uncomment the line below to test the full pipeline (requires API keys)
    # demonstrate_integration_with_pipeline()
