"""Test script to verify context fields work correctly with JudgeModule."""

import dspy
from src.factchecker import JudgeModule
from src.context_.context import openai_key

# Configure DSPy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key=openai_key))

# Initialize the judge module
judge = JudgeModule()

# Test 1: Without context fields (should work with defaults)
print("=" * 80)
print("Test 1: Without context fields")
print("=" * 80)
result1 = judge(statement="The Earth is flat.")
print(f"Statement: The Earth is flat.")
print(f"Verdict: {result1.overall_verdict}")
print(f"Confidence: {result1.confidence}")
print(f"Reasoning: {result1.reasoning}")
print()

# Test 2: With context fields
print("=" * 80)
print("Test 2: With context fields")
print("=" * 80)
statement = "Alaska Airlines has announced that it will launch new nonstop flights between Seattle and London Heathrow Airport on May 21, 2026."
topic = "Alaska Air"
date = "20251210"
source_urls = "https://www.cbsnews.com/news/joseph-emerson-alaska-airlines-pilot-flight-deck-audio-police-video/"

result2 = judge(
    statement=statement,
    topic=topic,
    date=date,
    source_urls=source_urls
)
print(f"Statement: {statement}")
print(f"Topic: {topic}")
print(f"Date: {date}")
print(f"Source URLs: {source_urls}")
print(f"Verdict: {result2.overall_verdict}")
print(f"Confidence: {result2.confidence}")
print(f"Reasoning: {result2.reasoning}")
print()

# Test 3: Test with dspy.Example (as used in optimization)
print("=" * 80)
print("Test 3: Using dspy.Example")
print("=" * 80)
example = dspy.Example(
    statement="The Moon landing was faked.",
    label="REFUTED",
    topic="Space Exploration",
    date="20240101",
    source_urls="https://example.com/moon-landing"
).with_inputs("statement", "topic", "date", "source_urls")

result3 = judge(
    statement=example.statement,
    topic=example.topic,
    date=example.date,
    source_urls=example.source_urls
)
print(f"Statement: {example.statement}")
print(f"Topic: {example.topic}")
print(f"Date: {example.date}")
print(f"Source URLs: {example.source_urls}")
print(f"Verdict: {result3.overall_verdict}")
print(f"Confidence: {result3.confidence}")
print(f"Reasoning: {result3.reasoning}")
print()

print("=" * 80)
print("All tests completed successfully!")
print("=" * 80)
