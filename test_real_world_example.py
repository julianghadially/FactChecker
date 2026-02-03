"""Real-world example demonstrating temporal awareness in action."""

from src.factchecker.simple.modules.judge_module import JudgeModule


def demonstrate_temporal_awareness():
    """
    Demonstrate how temporal awareness prevents false positives on recent claims.

    This example shows statements that might be incorrectly marked as SUPPORTED
    based on outdated training data, but will now trigger web search for verification.
    """

    print("=" * 80)
    print("REAL-WORLD TEMPORAL AWARENESS DEMONSTRATION")
    print("=" * 80)
    print("\nThis demonstrates how the enhanced JudgeModule automatically detects")
    print("time-sensitive claims that require web verification.\n")

    # Create judge module with web search disabled for this demo
    # (to show the detection logic without actually performing searches)
    judge = JudgeModule(enable_web_search=False)

    test_cases = [
        {
            "category": "Recent Political Events",
            "statement": "The 2025 presidential inauguration drew record crowds.",
            "risk": "LLM might confirm based on historical patterns, but actual 2025 data unknown"
        },
        {
            "category": "Technology Updates",
            "statement": "The latest iPhone model features a revolutionary camera system.",
            "risk": "LLM training data may contain info about earlier models"
        },
        {
            "category": "Sports Results",
            "statement": "The current NBA season MVP has outstanding statistics.",
            "risk": "LLM might guess based on previous seasons' patterns"
        },
        {
            "category": "Market Data",
            "statement": "Stock prices in January 2025 reached all-time highs.",
            "risk": "LLM can't know actual future/recent market data"
        },
        {
            "category": "Scientific Research",
            "statement": "Recent studies from 2024 show promising cancer treatment results.",
            "risk": "LLM training might not include latest research publications"
        },
        {
            "category": "Corporate News",
            "statement": "Tesla announced this year that it will open new factories.",
            "risk": "LLM might speculate based on past announcements"
        },
        {
            "category": "Climate Events",
            "statement": "Last month's temperatures broke several records worldwide.",
            "risk": "LLM can't know current weather/climate data"
        },
        {
            "category": "Safe - Historical Fact",
            "statement": "World War II ended in 1945.",
            "risk": "None - well-established historical fact"
        },
        {
            "category": "Safe - Scientific Constant",
            "statement": "The speed of light is approximately 299,792 km/s.",
            "risk": "None - physical constant doesn't change"
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['category']}")
        print("-" * 80)
        print(f"Statement: {case['statement']}")
        print(f"Risk: {case['risk']}")

        # Extract temporal references
        temporal_info = judge._extract_temporal_references(case['statement'])

        print(f"\nTemporal Analysis:")
        if temporal_info['dates']:
            print(f"  Dates detected: {len(temporal_info['dates'])}")
            for date in temporal_info['dates']:
                print(f"    - {date.strftime('%Y-%m-%d')}")
        else:
            print(f"  Dates detected: 0")

        if temporal_info['temporal_keywords']:
            print(f"  Temporal keywords: {temporal_info['temporal_keywords']}")
        else:
            print(f"  Temporal keywords: None")

        print(f"\nWeb Search Required: {'YES ✓' if temporal_info['needs_verification'] else 'NO ✗'}")

        if temporal_info['needs_verification']:
            print("  → This claim will be verified against current web sources")
        else:
            print("  → This claim can be safely evaluated using LLM knowledge")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nThe temporal awareness feature ensures that:")
    print("  1. Recent/current claims are ALWAYS verified against web sources")
    print("  2. Historical facts can be safely evaluated without web search")
    print("  3. False positives from outdated training data are prevented")
    print("  4. System automatically adapts as time passes (24-month rolling window)")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_temporal_awareness()
