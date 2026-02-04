"""Test script for SmartJudgeModule to verify routing logic."""

import dspy
from src.factchecker.modules import SmartJudgeModule


def test_smart_judge():
    """Test the SmartJudgeModule with different routing scenarios."""

    # Initialize the module
    print("Initializing SmartJudgeModule...")
    smart_judge = SmartJudgeModule(confidence_threshold=0.6)

    print("\n" + "="*80)
    print("TEST 1: Simple fact (should use JudgeModule)")
    print("="*80)
    statement1 = "Water boils at 100 degrees Celsius at sea level"
    result1 = smart_judge(statement=statement1)
    print(f"\nStatement: {result1.statement}")
    print(f"Verdict: {result1.overall_verdict}")
    print(f"Confidence: {result1.confidence}")
    print(f"Routing: {result1.routing_decision}")

    print("\n" + "="*80)
    print("TEST 2: Temporal claim (should route to FactCheckerPipeline)")
    print("="*80)
    statement2 = "In 2025, the global GDP growth rate exceeded 4%"
    result2 = smart_judge(statement=statement2)
    print(f"\nStatement: {result2.statement}")
    print(f"Verdict: {result2.overall_verdict}")
    print(f"Confidence: {result2.confidence}")
    print(f"Routing: {result2.routing_decision}")

    print("\n" + "="*80)
    print("TEST 3: With URLs provided (should pre-seed FactCheckerPipeline)")
    print("="*80)
    statement3 = "Python is a popular programming language"
    urls = ["https://www.python.org/about/"]
    result3 = smart_judge(statement=statement3, urls=urls)
    print(f"\nStatement: {result3.statement}")
    print(f"Verdict: {result3.overall_verdict}")
    print(f"Confidence: {result3.confidence}")
    print(f"Routing: {result3.routing_decision}")

    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)


if __name__ == "__main__":
    test_smart_judge()
