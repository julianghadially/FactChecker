"""Example usage and test cases for TemporalResearchRouterModule.

This file demonstrates the temporal detection capabilities of the router.
"""

from temporal_research_router_module import TemporalResearchRouterModule


def test_temporal_detection():
    """Test temporal signal detection with various claim types."""

    router = TemporalResearchRouterModule(max_page_visits=3)

    # Test cases with expected results
    test_claims = [
        # Very recent (should use "d" - daily)
        ("Apple just announced a new iPhone today", True, "d"),
        ("Breaking: Court has ruled on the case yesterday", True, "d"),
        ("This week the company opened a new store", True, "d"),

        # Recent (should use "w" - weekly)
        ("The government announced new regulations this month", True, "w"),
        ("Tesla recently launched a new feature", True, "w"),
        ("Last week the CEO resigned", True, "w"),

        # Temporal (should use "m" - monthly)
        ("Microsoft upgraded their cloud services in 2024", True, "m"),
        ("The Supreme Court ruled on the abortion case", True, "m"),
        ("Amazon has opened a new warehouse facility", True, "m"),
        ("The company announced quarterly earnings", True, "m"),

        # Non-temporal (should use regular search)
        ("The Earth orbits the Sun", False, ""),
        ("Paris is the capital of France", False, ""),
        ("Water boils at 100 degrees Celsius", False, ""),
        ("Shakespeare wrote Hamlet", False, ""),
    ]

    print("=" * 80)
    print("TEMPORAL DETECTION TEST RESULTS")
    print("=" * 80)

    for claim, expected_temporal, expected_recency in test_claims:
        is_temporal, recency = router._detect_temporal_signals(claim)

        status = "✓" if (is_temporal == expected_temporal and recency == expected_recency) else "✗"

        print(f"\n{status} Claim: {claim}")
        print(f"   Expected: temporal={expected_temporal}, recency='{expected_recency}'")
        print(f"   Detected: temporal={is_temporal}, recency='{recency}'")

        if is_temporal:
            # Test query enrichment
            test_query = "test query"
            enriched = router._enrich_query_with_temporal_context(test_query, claim, recency)
            print(f"   Query enrichment: '{test_query}' -> '{enriched}'")


def test_query_enrichment():
    """Test query enrichment with temporal context."""

    router = TemporalResearchRouterModule(max_page_visits=3)

    test_cases = [
        # (query, claim, recency, description)
        ("Apple iPhone", "Apple just announced iPhone", "d", "Very recent - add month/year"),
        ("Tesla stock", "Tesla stock price today", "d", "Very recent - add month/year"),
        ("Supreme Court ruling", "Court ruled last week", "w", "Recent - add year"),
        ("Amazon warehouse", "Amazon opened warehouse in 2024", "m", "Already has year"),
        ("Microsoft cloud 2024", "Microsoft upgraded cloud in 2024", "m", "Query already has year"),
    ]

    print("\n" + "=" * 80)
    print("QUERY ENRICHMENT TEST RESULTS")
    print("=" * 80)

    for query, claim, recency, description in test_cases:
        enriched = router._enrich_query_with_temporal_context(query, claim, recency)
        print(f"\n{description}")
        print(f"   Original: '{query}'")
        print(f"   Enriched: '{enriched}'")
        print(f"   Claim: {claim}")
        print(f"   Recency: {recency}")


if __name__ == "__main__":
    print("\n🔍 Testing Temporal Research Router Module\n")
    test_temporal_detection()
    test_query_enrichment()
    print("\n" + "=" * 80)
    print("✓ All tests completed!")
    print("=" * 80 + "\n")
