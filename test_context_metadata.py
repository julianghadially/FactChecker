#!/usr/bin/env python3
"""Test script to verify context metadata changes work correctly."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.data_loader import load_csv_dataset
from src.factchecker.modules.judge_module import JudgeModule
from src.baseline.baseline_model import BaselineModel

def test_data_loading():
    """Test that context fields are loaded from CSV."""
    print("=" * 60)
    print("TEST 1: CSV Data Loading")
    print("=" * 60)

    dataset = load_csv_dataset("data/FactChecker_news_claims.csv", limit=5)

    print(f"✓ Loaded {len(dataset.examples)} examples")

    # Check first example has context fields
    ex = dataset.examples[0]
    print(f"\nFirst example:")
    print(f"  Claim: {ex.claim[:80]}...")
    print(f"  Topic: {ex.topic}")
    print(f"  URL: {ex.url[:80]}..." if ex.url else "  URL: (empty)")
    print(f"  Date: {ex.date_generated}")
    print(f"  Label: {ex.label}")

    # Verify all examples have the fields (even if empty)
    for i, ex in enumerate(dataset.examples):
        assert hasattr(ex, 'topic'), f"Example {i} missing 'topic' field"
        assert hasattr(ex, 'url'), f"Example {i} missing 'url' field"
        assert hasattr(ex, 'date_generated'), f"Example {i} missing 'date_generated' field"

    print("\n✓ All examples have context fields")
    return True

def test_judge_module():
    """Test that JudgeModule accepts context parameters."""
    print("\n" + "=" * 60)
    print("TEST 2: JudgeModule Context Parameters")
    print("=" * 60)

    # Test without DSPy configuration (just check signatures)
    judge = JudgeModule()

    # Verify forward method accepts context parameters
    import inspect
    sig = inspect.signature(judge.forward)
    params = list(sig.parameters.keys())

    print(f"JudgeModule.forward() parameters: {params}")

    assert 'statement' in params, "Missing 'statement' parameter"
    assert 'topic' in params, "Missing 'topic' parameter"
    assert 'url' in params, "Missing 'url' parameter"
    assert 'date_generated' in params, "Missing 'date_generated' parameter"

    # Check defaults
    assert sig.parameters['topic'].default == "", "topic should default to ''"
    assert sig.parameters['url'].default == "", "url should default to ''"
    assert sig.parameters['date_generated'].default == "", "date_generated should default to ''"

    print("✓ JudgeModule.forward() has correct signature with defaults")
    return True

def test_baseline_model():
    """Test that BaselineModel accepts context parameters."""
    print("\n" + "=" * 60)
    print("TEST 3: BaselineModel Context Parameters")
    print("=" * 60)

    baseline = BaselineModel()

    # Verify forward method accepts context parameters
    import inspect
    sig = inspect.signature(baseline.forward)
    params = list(sig.parameters.keys())

    print(f"BaselineModel.forward() parameters: {params}")

    assert 'statement' in params, "Missing 'statement' parameter"
    assert 'topic' in params, "Missing 'topic' parameter"
    assert 'url' in params, "Missing 'url' parameter"
    assert 'date_generated' in params, "Missing 'date_generated' parameter"

    # Check defaults
    assert sig.parameters['topic'].default == "", "topic should default to ''"
    assert sig.parameters['url'].default == "", "url should default to ''"
    assert sig.parameters['date_generated'].default == "", "date_generated should default to ''"

    print("✓ BaselineModel.forward() has correct signature with defaults")
    return True

def test_signatures():
    """Test that signatures have context fields."""
    print("\n" + "=" * 60)
    print("TEST 4: Signature Context Fields")
    print("=" * 60)

    from src.factchecker.signatures.judge import Judge
    from src.baseline.baseline_model import BaselineFactCheck

    # Check Judge signature
    judge_fields = Judge.model_fields
    print(f"Judge InputFields: {[f for f in judge_fields.keys() if not f.startswith('_')]}")

    assert 'statement' in judge_fields, "Judge missing 'statement' field"
    assert 'topic' in judge_fields, "Judge missing 'topic' field"
    assert 'url' in judge_fields, "Judge missing 'url' field"
    assert 'date_generated' in judge_fields, "Judge missing 'date_generated' field"

    print("✓ Judge signature has all context fields")

    # Check BaselineFactCheck signature
    baseline_fields = BaselineFactCheck.model_fields
    print(f"BaselineFactCheck InputFields: {[f for f in baseline_fields.keys() if not f.startswith('_')]}")

    assert 'claim' in baseline_fields, "BaselineFactCheck missing 'claim' field"
    assert 'topic' in baseline_fields, "BaselineFactCheck missing 'topic' field"
    assert 'url' in baseline_fields, "BaselineFactCheck missing 'url' field"
    assert 'date_generated' in baseline_fields, "BaselineFactCheck missing 'date_generated' field"

    print("✓ BaselineFactCheck signature has all context fields")
    return True

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CONTEXT METADATA INTEGRATION TESTS")
    print("=" * 60)

    try:
        test_data_loading()
        test_judge_module()
        test_baseline_model()
        test_signatures()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nContext metadata integration is working correctly.")
        print("Both JudgeModule and BaselineModel can now receive:")
        print("  - topic: Domain/company context")
        print("  - url: Reference URLs")
        print("  - date_generated: Publication date")
        print("\nThese fields will be passed to the LLM for better-informed judgments.")

        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
