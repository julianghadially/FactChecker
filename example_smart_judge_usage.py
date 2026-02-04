"""Example usage of SmartJudgeModule demonstrating all routing scenarios."""

from src.factchecker.modules import SmartJudgeModule


def example_basic_usage():
    """Example 1: Basic usage with automatic routing."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage - High Confidence Historical Fact")
    print("="*80)

    smart_judge = SmartJudgeModule()

    statement = "The French Revolution began in 1789"
    result = smart_judge(statement=statement)

    print(f"\nStatement: {statement}")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Routing Decision: {result.routing_decision}")


def example_temporal_claim():
    """Example 2: Temporal claim detection triggering web research."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Temporal Claim - Automatic Web Research")
    print("="*80)

    smart_judge = SmartJudgeModule()

    statement = "In 2024, artificial intelligence capabilities surpassed human performance in most benchmarks"
    result = smart_judge(statement=statement)

    print(f"\nStatement: {statement}")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Routing Decision: {result.routing_decision}")
    if hasattr(result, 'claims'):
        print(f"Claims extracted: {len(result.claims)}")


def example_low_confidence_fallback():
    """Example 3: Low confidence triggering fallback to web research."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Low Confidence - Fallback to Web Research")
    print("="*80)

    smart_judge = SmartJudgeModule(confidence_threshold=0.6)

    # Obscure fact that LLM might not be confident about
    statement = "The Great Wall of China is visible from the International Space Station"
    result = smart_judge(statement=statement)

    print(f"\nStatement: {statement}")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Routing Decision: {result.routing_decision}")


def example_url_preseeding():
    """Example 4: URL pre-seeding with specific sources."""
    print("\n" + "="*80)
    print("EXAMPLE 4: URL Pre-Seeding - Using Provided Sources")
    print("="*80)

    smart_judge = SmartJudgeModule()

    statement = "DSPy is a framework for algorithmically optimizing LM prompts and weights"
    urls = ["https://github.com/stanfordnlp/dspy"]

    result = smart_judge(statement=statement, urls=urls)

    print(f"\nStatement: {statement}")
    print(f"URLs provided: {urls}")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Routing Decision: {result.routing_decision}")


def example_custom_threshold():
    """Example 5: Custom confidence threshold for more aggressive web research."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Threshold - More Aggressive Web Research")
    print("="*80)

    # Higher threshold means more fallback to web research
    smart_judge = SmartJudgeModule(confidence_threshold=0.8)

    statement = "Python is a programming language created by Guido van Rossum"
    result = smart_judge(statement=statement)

    print(f"\nStatement: {statement}")
    print(f"Confidence Threshold: 0.8 (higher = more aggressive)")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Routing Decision: {result.routing_decision}")


def example_unsupported_claim():
    """Example 6: Unsupported claim verdict triggering web research."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Unsupported Claim - Triggering Web Research")
    print("="*80)

    smart_judge = SmartJudgeModule()

    # Claim that requires verification
    statement = "There are exactly 42,186 species of spiders currently identified by scientists"
    result = smart_judge(statement=statement)

    print(f"\nStatement: {statement}")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Routing Decision: {result.routing_decision}")


def main():
    """Run all examples."""
    print("\n" + "#"*80)
    print("# SmartJudgeModule - Complete Usage Examples")
    print("#"*80)

    # Run examples
    example_basic_usage()
    example_temporal_claim()
    example_low_confidence_fallback()
    example_url_preseeding()
    example_custom_threshold()
    example_unsupported_claim()

    print("\n" + "#"*80)
    print("# All examples completed!")
    print("#"*80)


if __name__ == "__main__":
    main()
