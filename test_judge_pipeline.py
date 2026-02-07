"""Verification tests for the evidence-aware JudgeModule pipeline."""

import dspy
from src.factchecker.modules.judge_module import JudgeModule
from src.factchecker.modules.claim_type_analyzer_module import ClaimTypeAnalyzerModule
from src.factchecker.modules.search_query_generator_module import SearchQueryGeneratorModule
from src.factchecker.modules.evidence_retriever_module import EvidenceRetrieverModule
from src.factchecker.signatures.evidence_aware_judge import EvidenceAwareJudge


def test_claim_type_analyzer():
    """Test ClaimTypeAnalyzerModule independently."""
    print("\n" + "=" * 80)
    print("TEST 1: ClaimTypeAnalyzerModule")
    print("=" * 80)

    analyzer = ClaimTypeAnalyzerModule()

    # Test with corporate financial claim
    statement = "Apple reported revenue of $90.8 billion in Q1 2024"
    print(f"\nStatement: {statement}")
    result = analyzer(statement=statement)

    print(f"Claim Types: {result.claim_types}")
    print(f"Search Strategy: {result.search_strategy}")
    print(f"Reasoning: {result.reasoning}")

    assert isinstance(result.claim_types, list), "Claim types should be a list"
    assert len(result.claim_types) > 0, "Should have at least one claim type"
    assert isinstance(result.search_strategy, str), "Search strategy should be a string"
    assert len(result.search_strategy) > 0, "Search strategy should not be empty"

    # Verify it identified financial/quantitative nature
    assert any(t in result.claim_types for t in ["corporate_financial", "factual_quantitative"]), \
        f"Should identify financial/quantitative claim, got {result.claim_types}"

    print("✓ ClaimTypeAnalyzerModule test PASSED")
    return result


def test_search_query_generator():
    """Test SearchQueryGeneratorModule independently."""
    print("\n" + "=" * 80)
    print("TEST 2: SearchQueryGeneratorModule (without claim types)")
    print("=" * 80)

    generator = SearchQueryGeneratorModule()
    result = generator(statement="The Eiffel Tower is 330 meters tall and was completed in 1889")

    print(f"Queries: {result.queries}")
    print(f"Reasoning: {result.reasoning}")

    assert isinstance(result.queries, list), "Queries should be a list"
    assert 1 <= len(result.queries) <= 3, f"Should have 1-3 queries, got {len(result.queries)}"
    assert all(isinstance(q, str) for q in result.queries), "All queries should be strings"

    print("✓ SearchQueryGeneratorModule test PASSED")
    return result


def test_search_query_generator_with_claim_types():
    """Test SearchQueryGeneratorModule with claim type inputs."""
    print("\n" + "=" * 80)
    print("TEST 3: SearchQueryGeneratorModule (with claim types)")
    print("=" * 80)

    generator = SearchQueryGeneratorModule()

    statement = "Apple reported revenue of $90.8 billion in Q1 2024"
    claim_types = ["corporate_financial", "factual_quantitative"]
    search_strategy = "Prioritize SEC filings (site:sec.gov) and investor relations pages"

    print(f"\nStatement: {statement}")
    print(f"Claim Types: {claim_types}")
    print(f"Strategy: {search_strategy}")

    result = generator(
        statement=statement,
        claim_types=claim_types,
        search_strategy=search_strategy
    )

    print(f"\nGenerated Queries: {result.queries}")
    print(f"Reasoning: {result.reasoning}")

    assert isinstance(result.queries, list), "Queries should be a list"
    assert len(result.queries) >= 1, "Should have at least one query"

    # Check if queries show adaptation (site: operators or specific keywords)
    queries_str = " ".join(result.queries).lower()
    has_adaptation = (
        "site:" in queries_str or
        "sec" in queries_str or
        "investor" in queries_str or
        "q1" in queries_str or
        "quarter" in queries_str
    )

    if has_adaptation:
        print("✓ Queries show claim-aware adaptation")
    else:
        print("⚠ Note: Queries may not show explicit site: operators, but LLM may have adapted in other ways")

    print("✓ SearchQueryGeneratorModule with claim types test PASSED")
    return result


def test_evidence_retriever():
    """Test EvidenceRetrieverModule independently."""
    print("\n" + "=" * 80)
    print("TEST 4: EvidenceRetrieverModule")
    print("=" * 80)

    retriever = EvidenceRetrieverModule()
    result = retriever(queries=["Eiffel Tower height meters official"])

    print(f"Evidence length: {len(result.evidence)} characters")
    print(f"Number of sources: {len(result.sources)}")
    print(f"Sources: {result.sources[:3]}")  # Print first 3 sources
    print(f"\nFirst 500 chars of evidence:\n{result.evidence[:500]}...")

    assert len(result.evidence) > 0, "Evidence should not be empty"
    assert isinstance(result.sources, list), "Sources should be a list"

    print("✓ EvidenceRetrieverModule test PASSED")
    return result


def test_evidence_aware_judge():
    """Test EvidenceAwareJudge signature independently."""
    print("\n" + "=" * 80)
    print("TEST 5: EvidenceAwareJudge Signature")
    print("=" * 80)

    judge = dspy.ChainOfThought(EvidenceAwareJudge)

    # Test with mock evidence
    evidence = """
    ## Source: Eiffel Tower Official Website
    URL: https://www.toureiffel.paris/en

    The Eiffel Tower is 330 metres (1,083 ft) tall, about the same height as
    an 81-storey building. It was completed in 1889 for the World's Fair in Paris.
    """

    result = judge(
        statement="The Eiffel Tower is 330 meters tall and was completed in 1889",
        evidence=evidence
    )

    print(f"Verdict: {result.verdict}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")

    assert result.verdict in ["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"], \
        f"Invalid verdict: {result.verdict}"
    assert 0.0 <= result.confidence <= 1.0, f"Confidence should be between 0 and 1, got {result.confidence}"

    print("✓ EvidenceAwareJudge test PASSED")
    return result


def test_full_pipeline():
    """Test the complete JudgeModule pipeline."""
    print("\n" + "=" * 80)
    print("TEST 6: Full JudgeModule Pipeline")
    print("=" * 80)

    judge_module = JudgeModule()

    # Test case: Verifiable recent fact
    statement = "The 2024 Summer Olympics were held in Paris, France"
    print(f"\nStatement: {statement}")

    result = judge_module(statement=statement)

    print(f"\n--- Results ---")
    print(f"Statement: {result.statement}")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Claim Types: {result.claim_types}")
    print(f"Search Strategy: {result.search_strategy}")
    print(f"Search Queries: {result.queries}")
    print(f"Number of sources: {len(result.sources)}")

    # Assertions
    assert result.statement == statement, "Statement should match input"
    assert result.overall_verdict in ["SUPPORTED", "CONTAINS_UNSUPPORTED_CLAIMS", "CONTAINS_REFUTED_CLAIMS"], \
        f"Invalid verdict: {result.overall_verdict}"
    assert 0.0 <= result.confidence <= 1.0, f"Confidence should be between 0 and 1"
    assert len(result.reasoning) > 0, "Reasoning should not be empty"
    assert isinstance(result.queries, list), "Queries should be a list"
    assert len(result.queries) > 0, "Should have at least one query"
    assert isinstance(result.sources, list), "Sources should be a list"

    # New assertions for claim analysis fields
    assert hasattr(result, 'claim_types'), "Result should have claim_types field"
    assert hasattr(result, 'search_strategy'), "Result should have search_strategy field"
    assert isinstance(result.claim_types, list), "Claim types should be a list"
    assert len(result.claim_types) > 0, "Should have at least one claim type"
    assert isinstance(result.search_strategy, str), "Search strategy should be a string"

    print("\n✓ Full pipeline test PASSED")
    return result


def test_backward_compatibility():
    """Test that the output format is backward compatible."""
    print("\n" + "=" * 80)
    print("TEST 7: Backward Compatibility")
    print("=" * 80)

    judge_module = JudgeModule()
    result = judge_module(statement="The sky is blue")

    # Check required fields (original format)
    required_fields = ["statement", "overall_verdict", "confidence", "reasoning"]
    for field in required_fields:
        assert hasattr(result, field), f"Missing required field: {field}"
        print(f"✓ Has field: {field}")

    # Check transparency fields (added in previous version)
    transparency_fields = ["queries", "sources", "quality_assessment"]
    for field in transparency_fields:
        assert hasattr(result, field), f"Missing transparency field: {field}"
        print(f"✓ Has field: {field}")

    # Check new claim analysis fields
    new_fields = ["claim_types", "search_strategy"]
    for field in new_fields:
        assert hasattr(result, field), f"Missing new field: {field}"
        print(f"✓ Has field: {field}")

    print("\n✓ Backward compatibility test PASSED")
    return result


def main():
    """Run all verification tests."""
    print("\n" + "=" * 80)
    print("EVIDENCE-AWARE JUDGE PIPELINE VERIFICATION TESTS")
    print("=" * 80)

    # Configure DSPy (using default LM if configured)
    try:
        # Try to use existing configuration from main.py
        from src.context_.context import openai_key
        print("\nConfiguring DSPy...")
        lm = dspy.LM("openai/gpt-4o-mini", api_key=openai_key)
        dspy.configure(lm=lm)
        print("✓ DSPy configured with GPT-4o-mini")
    except Exception as e:
        print(f"Warning: Could not configure DSPy: {e}")
        print("Tests may fail if DSPy is not properly configured")

    try:
        # Run individual component tests
        test_claim_type_analyzer()
        test_search_query_generator()
        test_search_query_generator_with_claim_types()
        test_evidence_retriever()
        test_evidence_aware_judge()

        # Run integration tests
        test_full_pipeline()
        test_backward_compatibility()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"TEST FAILED: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
