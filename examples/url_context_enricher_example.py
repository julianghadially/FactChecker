"""Example usage of UrlContextEnricherModule for evidence-based fact checking."""

from src.factchecker.modules.url_context_enricher_module import UrlContextEnricherModule


def main():
    """Demonstrate URL context enrichment for fact checking."""

    # Initialize the module (automatically creates JudgeModule internally)
    enricher = UrlContextEnricherModule(
        max_urls=2,
        max_chars_per_url=1000
    )

    # Example 1: Statement without URLs (falls back to LLM knowledge)
    print("=" * 80)
    print("Example 1: Statement without URL context")
    print("=" * 80)
    statement1 = "The Eiffel Tower is located in Paris, France."
    result1 = enricher.forward(statement1)
    print(f"Statement: {result1.statement}")
    print(f"Verdict: {result1.overall_verdict}")
    print(f"Confidence: {result1.confidence}")
    print(f"Reasoning: {result1.reasoning}")
    print()

    # Example 2: Statement with single URL
    print("=" * 80)
    print("Example 2: Statement with URL context")
    print("=" * 80)
    statement2 = "OpenAI released GPT-4 in March 2023."
    url2 = "https://en.wikipedia.org/wiki/GPT-4"
    result2 = enricher.forward(statement2, url=url2)
    print(f"Statement: {result2.statement}")
    print(f"URL: {url2}")
    print(f"Verdict: {result2.overall_verdict}")
    print(f"Confidence: {result2.confidence}")
    print(f"Reasoning: {result2.reasoning}")
    print()

    # Example 3: Statement with multiple URLs
    print("=" * 80)
    print("Example 3: Statement with multiple URLs")
    print("=" * 80)
    statement3 = "Python 3.12 was released in October 2023."
    urls3 = [
        "https://www.python.org/downloads/release/python-3120/",
        "https://en.wikipedia.org/wiki/Python_(programming_language)"
    ]
    result3 = enricher.forward(statement3, urls=urls3)
    print(f"Statement: {result3.statement}")
    print(f"URLs: {', '.join(urls3)}")
    print(f"Verdict: {result3.overall_verdict}")
    print(f"Confidence: {result3.confidence}")
    print(f"Reasoning: {result3.reasoning}")
    print()


if __name__ == "__main__":
    main()
