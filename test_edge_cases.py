"""Test edge cases for ResearchModule - recent events and niche topics."""

import dspy
from src.context_.context import openai_key
from src.factchecker.modules import JudgeModule


def configure_dspy(model: str = "openai/gpt-4o-mini"):
    """Configure DSPy with the specified model."""
    lm = dspy.LM(model, api_key=openai_key)
    dspy.configure(lm=lm)
    dspy.configure_cache(
        enable_disk_cache=False,
        enable_memory_cache=False
    )


def test_statement(statement: str, description: str):
    """Test a statement with and without research."""
    print("\n" + "=" * 80)
    print(f"TEST: {description}")
    print("=" * 80)
    print(f"\nStatement: {statement}")

    configure_dspy()

    # Test without research
    judge_no_research = JudgeModule(use_research=False)
    print("\n[WITHOUT RESEARCH]")
    prediction_no = judge_no_research(statement=statement)
    print(f"  Verdict: {prediction_no.overall_verdict}")
    print(f"  Confidence: {prediction_no.confidence:.2f}")
    print(f"  Reasoning: {prediction_no.reasoning[:150]}...")

    # Test with research
    judge_with_research = JudgeModule(use_research=True)
    print("\n[WITH RESEARCH]")
    prediction_yes = judge_with_research(statement=statement)
    print(f"  Verdict: {prediction_yes.overall_verdict}")
    print(f"  Confidence: {prediction_yes.confidence:.2f}")
    print(f"  Reasoning: {prediction_yes.reasoning[:150]}...")

    if hasattr(prediction_yes, 'sources') and prediction_yes.sources:
        print(f"  Sources used: {len(prediction_yes.sources)}")
        for source in prediction_yes.sources[:2]:  # Show first 2
            print(f"    - {source['title'][:60]}...")

    print("\n" + "-" * 80)
    print("COMPARISON:")
    if prediction_no.overall_verdict != prediction_yes.overall_verdict:
        print(f"  ⚠️  Verdicts differ!")
        print(f"     Without research: {prediction_no.overall_verdict}")
        print(f"     With research: {prediction_yes.overall_verdict}")
    else:
        print(f"  ✓ Both agree: {prediction_no.overall_verdict}")

    if abs(prediction_no.confidence - prediction_yes.confidence) > 0.2:
        print(f"  ⚠️  Confidence differs significantly!")
        print(f"     Without research: {prediction_no.confidence:.2f}")
        print(f"     With research: {prediction_yes.confidence:.2f}")
    else:
        print(f"  ✓ Similar confidence levels")

    return prediction_no, prediction_yes


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# Edge Cases: Recent Events & Niche Topics")
    print("#" * 80)

    test_cases = [
        (
            "The 2024 Summer Olympics were held in Paris, France",
            "Recent Event (2024) - Within LLM knowledge"
        ),
        (
            "Taylor Swift's Eras Tour became the highest-grossing concert tour of all time in 2023",
            "Recent Event (2023) - Pop culture"
        ),
        (
            "OpenAI released GPT-5 in January 2025",
            "Recent False Claim - Should be refuted"
        ),
        (
            "The James Webb Space Telescope discovered signs of life on an exoplanet in 2024",
            "Niche Topic + Recent - Likely false but needs verification"
        ),
    ]

    results = []
    for statement, description in test_cases:
        try:
            result = test_statement(statement, description)
            results.append((description, result))
        except Exception as e:
            print(f"\n✗ Error testing '{description}': {str(e)}")

    print("\n" + "#" * 80)
    print("# Summary of All Tests")
    print("#" * 80)

    for description, (pred_no, pred_yes) in results:
        print(f"\n{description}")
        print(f"  Without research: {pred_no.overall_verdict} (conf: {pred_no.confidence:.2f})")
        print(f"  With research:    {pred_yes.overall_verdict} (conf: {pred_yes.confidence:.2f})")
        if pred_no.overall_verdict != pred_yes.overall_verdict:
            print(f"  → Research changed the verdict! ⚠️")

    print("\n" + "#" * 80)
    print("# All edge case tests completed!")
    print("#" * 80 + "\n")
