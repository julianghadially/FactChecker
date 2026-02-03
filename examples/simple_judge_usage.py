"""Simple usage examples for the enhanced JudgeModule.

This demonstrates how to use the JudgeModule with web search capability
in different scenarios.
"""

import dspy
from src.context_.context import openai_key
from src.factchecker.simple.modules.judge_module import JudgeModule


def example_1_basic_usage():
    """Example 1: Basic usage with web search enabled (default)."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage with Web Search")
    print("="*80)

    # Configure DSPy
    dspy.configure(lm=dspy.LM("openai/gpt-5-mini", api_key=openai_key))

    # Create judge (web search enabled by default)
    judge = JudgeModule()

    # Check a recent event
    statement = "Donald Trump won the 2024 U.S. presidential election."
    result = judge(statement=statement)

    print(f"\nStatement: {statement}")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Web Search Used: {result.web_search_performed}")
    print(f"Reasoning: {result.reasoning[:200]}...")


def example_2_historical_fact():
    """Example 2: Historical fact (should not trigger search)."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Historical Fact (No Search Needed)")
    print("="*80)

    dspy.configure(lm=dspy.LM("openai/gpt-5-mini", api_key=openai_key))
    judge = JudgeModule()

    statement = "World War II ended in 1945."
    result = judge(statement=statement)

    print(f"\nStatement: {statement}")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Web Search Used: {result.web_search_performed}")
    print(f"Note: Historical fact handled without web search")


def example_3_disable_search():
    """Example 3: Using JudgeModule without web search (original behavior)."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Disable Web Search (Original Behavior)")
    print("="*80)

    dspy.configure(lm=dspy.LM("openai/gpt-5-mini", api_key=openai_key))

    # Disable web search
    judge = JudgeModule(enable_web_search=False)

    statement = "SpaceX launched its first crewed mission to Mars in 2025."
    result = judge(statement=statement)

    print(f"\nStatement: {statement}")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Web Search Used: {result.web_search_performed}")
    print(f"Note: Without search, likely returns CONTAINS_UNSUPPORTED_CLAIMS")


def example_4_batch_evaluation():
    """Example 4: Batch evaluation of multiple statements."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Batch Evaluation")
    print("="*80)

    dspy.configure(lm=dspy.LM("openai/gpt-5-mini", api_key=openai_key))
    judge = JudgeModule(enable_web_search=True)

    statements = [
        "The capital of France is Paris.",
        "OpenAI released GPT-5 in early 2025.",
        "The Earth orbits around the Sun.",
    ]

    print("\nEvaluating multiple statements:\n")

    for i, stmt in enumerate(statements, 1):
        result = judge(statement=stmt)
        print(f"{i}. {stmt}")
        print(f"   Verdict: {result.overall_verdict}")
        print(f"   Search Used: {'Yes' if result.web_search_performed else 'No'}")
        print()


def example_5_integration_with_dspy_evaluate():
    """Example 5: Integration with DSPy's evaluation framework."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Integration with DSPy Evaluate")
    print("="*80)

    dspy.configure(lm=dspy.LM("openai/gpt-5-mini", api_key=openai_key))

    # Create examples
    examples = [
        dspy.Example(
            statement="The Great Wall of China is visible from space.",
            label="REFUTED"
        ).with_inputs("statement"),
        dspy.Example(
            statement="Water freezes at 0 degrees Celsius at standard pressure.",
            label="SUPPORTED"
        ).with_inputs("statement"),
    ]

    # Define metric
    def accuracy_metric(example, prediction) -> float:
        pred_label = prediction.overall_verdict if hasattr(prediction, 'overall_verdict') else str(prediction)
        return 1.0 if example.label == pred_label else 0.0

    # Create evaluator
    judge = JudgeModule(enable_web_search=True)
    evaluator = dspy.Evaluate(
        devset=examples,
        metric=accuracy_metric,
        num_threads=1,
        display_progress=True
    )

    # Run evaluation
    print("\nRunning evaluation...")
    result = evaluator(judge)
    print(f"Average Score: {result:.2%}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("ENHANCED JUDGEMODULE - USAGE EXAMPLES")
    print("="*80)

    try:
        example_1_basic_usage()
        example_2_historical_fact()
        example_3_disable_search()
        example_4_batch_evaluation()
        example_5_integration_with_dspy_evaluate()

        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have:")
        print("1. Set OPENAI_AGENTJUDGEJG_KEY environment variable")
        print("2. Set SERPER_KEY environment variable")
        print("3. Installed required packages: dspy, requests")


if __name__ == "__main__":
    main()
