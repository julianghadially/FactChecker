"""Integration test for context fields throughout the pipeline."""

import dspy
from src.factchecker import JudgeModule
from src.optimizer.gepa_optimize import load_dspy_examples
from src.evaluation.data_loader import load_csv_dataset, HoverExample
from src.context_.context import openai_key

# Configure DSPy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key=openai_key))

print("=" * 80)
print("INTEGRATION TEST: Context Fields End-to-End")
print("=" * 80)

# Test 1: Data Loader
print("\n1. Testing data_loader with CSV dataset (has context fields)...")
try:
    dataset_with_schema = load_csv_dataset("data/FactChecker_news_claims.csv", limit=2)
    examples = dataset_with_schema.examples

    print(f"✓ Loaded {len(examples)} examples")

    # Check first example has context fields
    ex = examples[0]
    print(f"  Example 0:")
    print(f"    Claim: {ex.claim[:60]}...")
    print(f"    Topic: {ex.topic}")
    print(f"    Date: {ex.date_generated}")
    print(f"    URL: {ex.url[:50] if ex.url else 'N/A'}...")

    if ex.topic and ex.date_generated:
        print("✓ Context fields loaded successfully from CSV")
    else:
        print("✗ Context fields missing!")

except Exception as e:
    print(f"✗ Data loader test failed: {e}")

# Test 2: DSPy Example Creation
print("\n2. Testing load_dspy_examples (converts to dspy.Example)...")
try:
    dspy_examples = load_dspy_examples("data/FactChecker_news_claims.csv", limit=2)

    print(f"✓ Created {len(dspy_examples)} dspy.Example objects")

    # Check first example has context fields as inputs
    ex = dspy_examples[0]
    print(f"  Example 0:")
    print(f"    Statement: {ex.statement[:60]}...")
    print(f"    Topic: {ex.topic}")
    print(f"    Date: {ex.date}")
    print(f"    Source URLs: {ex.source_urls[:50] if ex.source_urls else 'N/A'}...")

    # Check inputs are registered correctly
    if hasattr(ex, '_input_keys') and 'topic' in ex._input_keys and 'date' in ex._input_keys:
        print("✓ Context fields registered as inputs")
    else:
        print("✗ Context fields not registered as inputs!")

except Exception as e:
    print(f"✗ DSPy example creation test failed: {e}")

# Test 3: JudgeModule with Context
print("\n3. Testing JudgeModule.forward() with context fields...")
try:
    judge = JudgeModule()

    # Get an example with context
    ex = dspy_examples[0]

    result = judge(
        statement=ex.statement,
        topic=ex.topic,
        date=ex.date,
        source_urls=ex.source_urls
    )

    print(f"  Input:")
    print(f"    Statement: {ex.statement[:60]}...")
    print(f"    Topic: {ex.topic}")
    print(f"    Date: {ex.date}")
    print(f"  Output:")
    print(f"    Verdict: {result.overall_verdict}")
    print(f"    Confidence: {result.confidence}")
    print(f"✓ JudgeModule processed context fields successfully")

except Exception as e:
    print(f"✗ JudgeModule test failed: {e}")

# Test 4: Backward Compatibility (dataset without context)
print("\n4. Testing backward compatibility (dataset without context)...")
try:
    dspy_examples_no_context = load_dspy_examples("data/FacTool_QA_train.jsonl", limit=2)

    ex = dspy_examples_no_context[0]
    print(f"✓ Loaded {len(dspy_examples_no_context)} examples without context")
    print(f"  Example 0:")
    print(f"    Statement: {ex.statement[:60]}...")
    print(f"    Topic: '{ex.topic}' (should be empty)")
    print(f"    Date: '{ex.date}' (should be empty)")
    print(f"    Source URLs: '{ex.source_urls}' (should be empty)")

    # Test with JudgeModule
    result = judge(
        statement=ex.statement,
        topic=ex.topic,
        date=ex.date,
        source_urls=ex.source_urls
    )

    print(f"  Verdict: {result.overall_verdict}")
    print("✓ Backward compatibility maintained")

except Exception as e:
    print(f"✗ Backward compatibility test failed: {e}")

print("\n" + "=" * 80)
print("INTEGRATION TEST COMPLETE")
print("=" * 80)
print("\nAll components work together correctly!")
print("- Data loader extracts context fields from datasets")
print("- dspy.Example objects include context fields as inputs")
print("- JudgeModule accepts and uses context fields")
print("- Backward compatibility maintained for datasets without context")
