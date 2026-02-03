"""Test script to demonstrate temporal awareness in JudgeModule."""

from datetime import datetime
from src.factchecker.simple.modules.judge_module import JudgeModule


def test_temporal_extraction():
    """Test the temporal reference extraction functionality."""

    # Create a judge module instance (we don't need web search enabled for this test)
    judge = JudgeModule(enable_web_search=False)

    # Test cases with various temporal patterns
    test_statements = [
        # Date patterns
        "The event happened on 2024-06-15.",
        "The company was founded in January 2025.",
        "The product launched in 2024.",
        "This occurred on March 2023.",

        # Temporal keywords
        "The recent study shows that climate change is accelerating.",
        "The current president announced new policies.",
        "The latest data indicates strong economic growth.",
        "This year's budget increased by 10%.",
        "Last month's sales were record-breaking.",

        # Mixed patterns
        "In 2025, the latest research confirms previous findings.",
        "Recent developments in January 2024 changed everything.",

        # Old dates (should NOT trigger web search)
        "World War II ended in 1945.",
        "The internet was invented in the 1980s.",
        "Shakespeare wrote Hamlet in 1600.",

        # No temporal references
        "Water boils at 100 degrees Celsius.",
        "Paris is the capital of France.",
    ]

    print("=" * 80)
    print("TEMPORAL AWARENESS TEST")
    print("=" * 80)
    print(f"\nToday's date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Cutoff date (24 months ago): {(datetime.now().year - 2)}-{datetime.now().month:02d}")
    print("\n" + "=" * 80)

    for i, statement in enumerate(test_statements, 1):
        print(f"\n{i}. Statement: {statement}")
        print("-" * 80)

        result = judge._extract_temporal_references(statement)

        print(f"   Dates found: {len(result['dates'])}")
        for date in result['dates']:
            print(f"     - {date.strftime('%Y-%m-%d')}")

        print(f"   Temporal keywords: {len(result['temporal_keywords'])}")
        for keyword in result['temporal_keywords']:
            print(f"     - '{keyword}'")

        print(f"   Needs verification: {result['needs_verification']}")

        if result['needs_verification']:
            print("   ✓ This statement WILL trigger web search")
        else:
            print("   ✗ This statement will NOT trigger web search")


def test_knowledge_limitation_detection():
    """Test the enhanced knowledge limitation detection."""

    judge = JudgeModule(enable_web_search=False)

    print("\n\n" + "=" * 80)
    print("KNOWLEDGE LIMITATION DETECTION TEST")
    print("=" * 80)

    test_cases = [
        {
            "statement": "The recent elections in 2025 resulted in a change of government.",
            "reasoning": "Based on my training data, I can confirm this information.",
            "verdict": "SUPPORTED",
            "expected": True,  # Should trigger due to temporal references
        },
        {
            "statement": "Water boils at 100 degrees Celsius at sea level.",
            "reasoning": "This is a well-established scientific fact.",
            "verdict": "SUPPORTED",
            "expected": False,  # Should NOT trigger
        },
        {
            "statement": "The latest smartphone model has excellent features.",
            "reasoning": "I cannot verify current product specifications.",
            "verdict": "CONTAINS_UNSUPPORTED_CLAIMS",
            "expected": True,  # Should trigger due to verdict AND temporal keyword
        },
        {
            "statement": "Napoleon was defeated at Waterloo.",
            "reasoning": "This is a well-documented historical fact.",
            "verdict": "SUPPORTED",
            "expected": False,  # Should NOT trigger
        },
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Statement: {case['statement']}")
        print(f"   Verdict: {case['verdict']}")
        print(f"   Reasoning: {case['reasoning']}")
        print("-" * 80)

        result = judge._detect_knowledge_limitations(
            case['reasoning'],
            case['verdict'],
            case['statement']
        )

        print(f"   Detection result: {result}")
        print(f"   Expected: {case['expected']}")

        if result == case['expected']:
            print("   ✓ PASS")
        else:
            print("   ✗ FAIL")


if __name__ == "__main__":
    test_temporal_extraction()
    test_knowledge_limitation_detection()
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)
