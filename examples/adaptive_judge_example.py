"""Example usage of AdaptiveJudgeModule.

This script demonstrates the adaptive fact-checking module that intelligently
routes between fast JudgeModule and thorough FactCheckerPipeline based on
confidence levels.
"""

import dspy
import os
from src.factchecker.modules.adaptive_judge_module import AdaptiveJudgeModule

# Configure DSPy with your LLM
# Adjust model and API key as needed
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
dspy.configure(lm=lm)


def test_adaptive_judge():
    """Test the adaptive judge module with various statements."""

    # Initialize with default settings (threshold=0.7, fallback enabled)
    adaptive_judge = AdaptiveJudgeModule(
        confidence_threshold=0.7,
        enable_fallback=True,
        max_judge_iterations=3,
        max_page_visits=3
    )

    # Test cases
    test_statements = [
        # Case 1: Well-known fact - should be confident, no fallback
        "The Earth orbits around the Sun.",

        # Case 2: Obviously false - should be confident, no fallback
        "The Moon is made of cheese.",

        # Case 3: Obscure/recent fact - likely low confidence, triggers fallback
        "The latest SpaceX Starship test flight in December 2024 achieved orbital velocity.",

        # Case 4: Uncertain claim - likely triggers fallback
        "Company X's Q3 2024 revenue exceeded $500 million for the first time.",
    ]

    print("=" * 80)
    print("ADAPTIVE JUDGE MODULE DEMO")
    print("=" * 80)

    for i, statement in enumerate(test_statements, 1):
        print(f"\n{'='*80}")
        print(f"Test Case {i}: {statement}")
        print(f"{'='*80}\n")

        result = adaptive_judge(statement=statement)

        print(f"Verdict: {result.overall_verdict}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Fallback Triggered: {result.fallback_triggered}")
        print(f"\nReasoning:\n{result.reasoning}")

        if result.fallback_triggered:
            print(f"\n📊 Pipeline Details:")
            print(f"   Claims Extracted: {len(result.claims)}")
            print(f"   Claims: {result.claims}")
            if hasattr(result, 'claim_results') and result.claim_results:
                print(f"\n   Claim-Level Results:")
                for j, claim_result in enumerate(result.claim_results, 1):
                    print(f"      {j}. {claim_result.claim}")
                    print(f"         Verdict: {claim_result.verdict}")
                    print(f"         Evidence: {claim_result.evidence_summary[:200]}...")

        print()


def test_without_fallback():
    """Test the module with fallback disabled."""

    print("\n" + "=" * 80)
    print("TESTING WITH FALLBACK DISABLED")
    print("=" * 80)

    # Initialize with fallback disabled
    judge_only = AdaptiveJudgeModule(
        confidence_threshold=0.7,
        enable_fallback=False  # Fallback disabled
    )

    statement = "The latest SpaceX Starship test flight in December 2024 achieved orbital velocity."
    print(f"\nStatement: {statement}\n")

    result = judge_only(statement=statement)

    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Fallback Triggered: {result.fallback_triggered}")
    print(f"\nReasoning:\n{result.reasoning}")


def test_custom_threshold():
    """Test with different confidence thresholds."""

    print("\n" + "=" * 80)
    print("TESTING WITH CUSTOM THRESHOLD (0.5)")
    print("=" * 80)

    # Lower threshold means less likely to trigger fallback
    adaptive_judge = AdaptiveJudgeModule(
        confidence_threshold=0.5,  # Lower threshold
        enable_fallback=True,
        max_judge_iterations=2,
        max_page_visits=2
    )

    statement = "The latest SpaceX Starship test flight in December 2024 achieved orbital velocity."
    print(f"\nStatement: {statement}\n")

    result = adaptive_judge(statement=statement)

    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Fallback Triggered: {result.fallback_triggered}")
    print(f"\nReasoning:\n{result.reasoning}")


if __name__ == "__main__":
    # Ensure API keys are set
    required_keys = ['OPENAI_API_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        print(f"❌ Error: Missing required environment variables: {', '.join(missing_keys)}")
        print("\nPlease set:")
        for key in missing_keys:
            print(f"  export {key}='your-api-key'")
        exit(1)

    # Note: Serper and Firecrawl keys only needed if fallback is triggered
    optional_keys = ['SERPER_API_KEY', 'FIRECRAWL_API_KEY']
    missing_optional = [key for key in optional_keys if not os.getenv(key)]

    if missing_optional:
        print(f"⚠️  Warning: Optional keys not set: {', '.join(missing_optional)}")
        print("   These are only needed if fallback to FactCheckerPipeline is triggered.\n")

    # Run tests
    try:
        test_adaptive_judge()
        test_without_fallback()
        test_custom_threshold()

        print("\n" + "=" * 80)
        print("✅ All tests completed!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        raise
