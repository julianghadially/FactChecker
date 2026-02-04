"""Test script to verify the multi-query search enhancement."""

import dspy
from src.factchecker.simple.modules.judge_module import JudgeModule
from src.factchecker.simple.signatures.query_generator import QueryGenerator


def test_query_generator():
    """Test that QueryGenerator creates focused search queries."""
    print("Testing QueryGenerator...")
    print("-" * 80)

    # Configure DSPy with a model (you may need to set this up based on your environment)
    try:
        query_gen = dspy.ChainOfThought(QueryGenerator)

        test_statement = (
            "Mondelez has been selling sugar-free Oreo cookies in the United States "
            "for several years prior to the announced Oreo Zero Sugar launch"
        )

        print(f"Statement: {test_statement}\n")

        result = query_gen(statement=test_statement)

        print(f"Generated queries ({len(result.queries)}):")
        for i, query in enumerate(result.queries, 1):
            print(f"  {i}. {query}")

        print("\n✓ QueryGenerator test completed successfully!")
        return True

    except Exception as e:
        print(f"✗ QueryGenerator test failed: {e}")
        return False


def test_judge_module_integration():
    """Test that JudgeModule integrates the multi-query enhancement."""
    print("\nTesting JudgeModule integration...")
    print("-" * 80)

    try:
        judge_module = JudgeModule()

        # Verify that query_generator is initialized
        assert hasattr(judge_module, 'query_generator'), "query_generator not initialized"
        assert judge_module.query_generator is not None, "query_generator is None"

        print("✓ JudgeModule has query_generator attribute")
        print("✓ JudgeModule integration test passed!")
        return True

    except Exception as e:
        print(f"✗ JudgeModule integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("Multi-Query Search Enhancement Test Suite")
    print("=" * 80)

    results = []

    # Test 1: Query Generator
    results.append(("QueryGenerator", test_query_generator()))

    # Test 2: JudgeModule Integration
    results.append(("JudgeModule Integration", test_judge_module_integration()))

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)
    print("\n" + ("All tests passed! 🎉" if all_passed else "Some tests failed ❌"))

    return all_passed


if __name__ == "__main__":
    main()
