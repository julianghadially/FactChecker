"""Test script to demonstrate the QueryGenerator signature."""

import dspy
from src.factchecker.simple.signatures.query_generator import QueryGenerator


def test_query_generator():
    """Test the QueryGenerator with various statements."""

    # Configure DSPy with a simple LLM (you'll need to set up your LLM)
    query_gen = dspy.ChainOfThought(QueryGenerator)

    # Test cases covering different types of statements
    test_statements = [
        "Deutsche Bank upgraded 3M to buy in January 2025",
        "Apple released iPhone 15 in September 2023",
        "Elon Musk acquired Twitter for $44 billion in October 2022",
        "The Federal Reserve raised interest rates by 0.25% in March 2024",
        "Microsoft announced a partnership with OpenAI worth $10 billion",
    ]

    print("=" * 80)
    print("Testing QueryGenerator - Intelligent Query Optimization")
    print("=" * 80)

    for i, statement in enumerate(test_statements, 1):
        print(f"\n\n{'=' * 80}")
        print(f"TEST {i}")
        print('=' * 80)
        print(f"📝 ORIGINAL STATEMENT:\n{statement}")
        print()

        try:
            result = query_gen(statement=statement)

            print(f"🧠 REASONING:\n{result.reasoning}")
            print()
            print(f"🔍 GENERATED QUERIES ({len(result.queries)}):")
            for j, query in enumerate(result.queries, 1):
                print(f"   {j}. {query}")

        except Exception as e:
            print(f"❌ ERROR: {e}")

    print("\n\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_query_generator()
