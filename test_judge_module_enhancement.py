"""Test script for enhanced JudgeModule with URL support."""

import dspy
from src.factchecker.simple.modules.judge_module import JudgeModule

# Configure dspy with a default LLM (you may need to adjust this based on your setup)
# Uncomment and configure if needed:
# lm = dspy.LM(model='openai/gpt-4o-mini')
# dspy.configure(lm=lm)

def test_without_url():
    """Test the judge module without URL (backward compatibility)."""
    print("\n" + "="*80)
    print("TEST 1: Without URL (backward compatibility)")
    print("="*80)

    judge = JudgeModule()
    statement = "The United States has the highest number of nuclear power plants in the world"

    print(f"Statement: {statement}")
    print("URL: None")
    print("\nProcessing...")

    result = judge.forward(statement=statement)

    print(f"\nResults:")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")

def test_with_url():
    """Test the judge module with a URL."""
    print("\n" + "="*80)
    print("TEST 2: With URL (evidence-based verification)")
    print("="*80)

    judge = JudgeModule()
    statement = "Alaska Airlines is launching nonstop flights from Seattle to London in May 2026"
    url = "https://thepointsguy.com/news/alaska-airlines-london-heathrow-seattle-nonstop-flights/"

    print(f"Statement: {statement}")
    print(f"URL: {url}")
    print("\nProcessing (this may take a moment to scrape the URL)...")

    result = judge.forward(statement=statement, url=url)

    print(f"\nResults:")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")

def test_with_multiple_urls():
    """Test the judge module with multiple comma-separated URLs."""
    print("\n" + "="*80)
    print("TEST 3: With Multiple URLs")
    print("="*80)

    judge = JudgeModule()
    statement = "Alaska Airlines is launching nonstop flights from Seattle to London"
    urls = "https://thepointsguy.com/news/alaska-airlines-london-heathrow-seattle-nonstop-flights/,https://www.bizjournals.com/seattle/news/2025/12/09/alaska-airlines-start-date-london-heathrow-airport.html"

    print(f"Statement: {statement}")
    print(f"URLs: {urls}")
    print("\nProcessing (this may take a moment to scrape the URLs)...")

    result = judge.forward(statement=statement, url=urls)

    print(f"\nResults:")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")

def test_with_invalid_url():
    """Test graceful handling of invalid URL."""
    print("\n" + "="*80)
    print("TEST 4: With Invalid URL (graceful failure)")
    print("="*80)

    judge = JudgeModule()
    statement = "The sky is blue"
    url = "https://this-url-definitely-does-not-exist-12345.com"

    print(f"Statement: {statement}")
    print(f"URL: {url}")
    print("\nProcessing...")

    result = judge.forward(statement=statement, url=url)

    print(f"\nResults:")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
    print("\nNote: Should fall back to knowledge-only judgment when URL scraping fails")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("ENHANCED JUDGEMODULE TESTS")
    print("="*80)
    print("\nThese tests verify the JudgeModule can:")
    print("1. Work without URLs (backward compatibility)")
    print("2. Accept and use evidence from a single URL")
    print("3. Accept and use evidence from multiple URLs")
    print("4. Handle scraping failures gracefully")

    # Run tests
    # Note: You may want to run these selectively to avoid too many API calls

    try:
        test_without_url()
    except Exception as e:
        print(f"\nTest 1 failed with error: {e}")

    # Uncomment the tests you want to run:
    # try:
    #     test_with_url()
    # except Exception as e:
    #     print(f"\nTest 2 failed with error: {e}")

    # try:
    #     test_with_multiple_urls()
    # except Exception as e:
    #     print(f"\nTest 3 failed with error: {e}")

    # try:
    #     test_with_invalid_url()
    # except Exception as e:
    #     print(f"\nTest 4 failed with error: {e}")

    print("\n" + "="*80)
    print("TESTS COMPLETE")
    print("="*80)
