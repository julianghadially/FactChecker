"""Test script for ResearchModule and integrated JudgeModule."""

import dspy
from src.context_.context import openai_key
from src.factchecker.modules import ResearchModule, JudgeModule


def configure_dspy(model: str = "openai/gpt-4o-mini"):
    """Configure DSPy with the specified model."""
    lm = dspy.LM(model, api_key=openai_key)
    dspy.configure(lm=lm)
    dspy.configure_cache(
        enable_disk_cache=False,
        enable_memory_cache=False
    )


def test_research_module():
    """Test ResearchModule standalone functionality."""
    print("\n" + "=" * 80)
    print("TEST 1: ResearchModule Standalone")
    print("=" * 80)

    configure_dspy()
    research = ResearchModule(num_queries=2, num_sources=3)

    # Test with a recent event (2024 Olympics)
    statement = "The 2024 Summer Olympics were held in Paris, France"

    print(f"\nStatement: {statement}")
    print("\nSearching and scraping web sources...")

    result = research(statement=statement)

    if result.success:
        print(f"\n✓ Research successful!")
        print(f"\nGenerated queries:")
        for i, query in enumerate(result.search_queries, 1):
            print(f"  {i}. {query}")

        print(f"\nSources found: {len(result.sources)}")
        for i, source in enumerate(result.sources, 1):
            print(f"  {i}. {source['title']}")
            print(f"     URL: {source['url']}")
            print(f"     Snippet: {source['snippet'][:100]}...")

        print(f"\nEvidence Summary:")
        print(f"  {result.evidence_summary[:500]}...")
    else:
        print(f"\n✗ Research failed: {result.error}")

    return result


def test_judge_module_without_research():
    """Test JudgeModule without research (LLM knowledge only)."""
    print("\n" + "=" * 80)
    print("TEST 2: JudgeModule WITHOUT Research")
    print("=" * 80)

    configure_dspy()
    judge = JudgeModule(use_research=False)

    statement = "The 2024 Summer Olympics were held in Paris, France"

    print(f"\nStatement: {statement}")
    print("\nJudging without web research...")

    prediction = judge(statement=statement)

    print(f"\n✓ Verdict: {prediction.overall_verdict}")
    print(f"✓ Confidence: {prediction.confidence:.2f}")
    print(f"\nReasoning:")
    print(f"  {prediction.reasoning}")

    return prediction


def test_judge_module_with_research():
    """Test JudgeModule with research enabled."""
    print("\n" + "=" * 80)
    print("TEST 3: JudgeModule WITH Research")
    print("=" * 80)

    configure_dspy()
    judge = JudgeModule(use_research=True)

    statement = "The 2024 Summer Olympics were held in Paris, France"

    print(f"\nStatement: {statement}")
    print("\nJudging with web research enabled...")

    prediction = judge(statement=statement)

    print(f"\n✓ Verdict: {prediction.overall_verdict}")
    print(f"✓ Confidence: {prediction.confidence:.2f}")
    print(f"\nReasoning:")
    print(f"  {prediction.reasoning}")

    if hasattr(prediction, 'sources') and prediction.sources:
        print(f"\nSources used ({len(prediction.sources)}):")
        for i, source in enumerate(prediction.sources, 1):
            print(f"  {i}. {source['title']}")
            print(f"     {source['url']}")

    if hasattr(prediction, 'evidence') and prediction.evidence:
        print(f"\nEvidence summary:")
        print(f"  {prediction.evidence[:300]}...")

    return prediction


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# Testing ResearchModule and JudgeModule Integration")
    print("#" * 80)

    # Test 1: ResearchModule standalone
    research_result = test_research_module()

    # Test 2: JudgeModule without research
    prediction_no_research = test_judge_module_without_research()

    # Test 3: JudgeModule with research
    prediction_with_research = test_judge_module_with_research()

    print("\n" + "=" * 80)
    print("COMPARISON: With vs Without Research")
    print("=" * 80)
    print(f"\nWithout research:")
    print(f"  Verdict: {prediction_no_research.overall_verdict}")
    print(f"  Confidence: {prediction_no_research.confidence:.2f}")

    print(f"\nWith research:")
    print(f"  Verdict: {prediction_with_research.overall_verdict}")
    print(f"  Confidence: {prediction_with_research.confidence:.2f}")

    print("\n" + "#" * 80)
    print("# All tests completed!")
    print("#" * 80 + "\n")
