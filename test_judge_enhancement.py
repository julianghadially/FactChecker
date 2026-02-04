"""Test script to demonstrate the enhanced JudgeModule with web search fallback."""

from src.factchecker.simple.modules.judge_module import JudgeModule


def test_judge_module():
    """Test the enhanced JudgeModule with various statements."""

    judge = JudgeModule()

    # Test cases covering different scenarios
    test_statements = [
        # Recent event (post-2024) - should trigger web search
        "Donald Trump won the 2024 U.S. presidential election",

        # Historical fact - should NOT trigger web search
        "The United States declared independence in 1776",

        # Recent tech news - might trigger web search
        "OpenAI released GPT-5 in 2025",
    ]

    print("=" * 80)
    print("Testing Enhanced JudgeModule with Web Search Fallback")
    print("=" * 80)

    for i, statement in enumerate(test_statements, 1):
        print(f"\n\n{'=' * 80}")
        print(f"TEST {i}: {statement}")
        print('=' * 80)

        result = judge.forward(statement, web_search_enabled=True)

        print(f"\n📊 VERDICT: {result.overall_verdict}")
        print(f"🎯 CONFIDENCE: {result.confidence}")
        print(f"🌐 USED WEB SEARCH: {result.used_web_search}")
        print(f"\n💭 REASONING:\n{result.reasoning}")

        if result.used_web_search and result.evidence:
            print(f"\n📚 EVIDENCE (truncated):\n{result.evidence[:500]}...")

    print("\n\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_judge_module()
