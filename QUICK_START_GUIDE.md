# Quick Start Guide: Multi-Query Enhanced Judge Module

## Basic Usage

The JudgeModule now automatically uses multi-query search when needed. No changes required to existing code!

### Example 1: Simple Statement Evaluation

```python
from src.factchecker.simple.modules.judge_module import JudgeModule

# Initialize the module
judge = JudgeModule()

# Evaluate a statement
statement = "Apple released the iPhone 15 in September 2023"
result = judge(statement=statement)

# Access results
print(f"Verdict: {result.overall_verdict}")
print(f"Confidence: {result.confidence}")
print(f"Research triggered: {result.research_triggered}")
print(f"Reasoning: {result.reasoning}")
```

### Example 2: Temporal Claim (Multi-Query Shines Here!)

```python
from src.factchecker.simple.modules.judge_module import JudgeModule

judge = JudgeModule()

# Statement with temporal claim
statement = (
    "Mondelez has been selling sugar-free Oreo cookies in the United States "
    "for several years prior to the announced Oreo Zero Sugar launch"
)

result = judge(statement=statement)

# The multi-query enhancement will:
# 1. Generate focused queries about:
#    - Oreo Zero Sugar launch date
#    - Sugar-free Oreo history in US
#    - Mondelez product timeline
# 2. Search all queries and deduplicate results
# 3. Scrape 3-4 sources
# 4. Make informed verdict with evidence

print(f"Verdict: {result.overall_verdict}")  # More likely to be REFUTED
print(f"Confidence: {result.confidence}")     # Higher confidence
```

## Output Format

```python
dspy.Prediction(
    statement="The evaluated statement",
    overall_verdict="SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS",
    confidence=0.85,  # Float between 0.0 and 1.0
    reasoning="Explanation of the verdict with evidence citations",
    research_triggered=True  # Whether web research was used
)
```

## When Multi-Query Search Triggers

Web research (including multi-query search) is triggered when:

1. **Low confidence CONTAINS_UNSUPPORTED_CLAIMS:**
   - First pass verdict is CONTAINS_UNSUPPORTED_CLAIMS
   - AND confidence < 0.6

2. **Knowledge limitations detected:**
   - Reasoning mentions: 'knowledge cutoff', 'cannot verify', 'cannot confirm', '2024', '2025'

## Comparing Before/After Behavior

### Before Enhancement

```python
statement = "Company X launched Product Y in 2025"

# Single query: "Company X launched Product Y in 2025"
# Top 2 results scraped
# May miss specific evidence about launch date
```

### After Enhancement

```python
statement = "Company X launched Product Y in 2025"

# Multi-query generation produces:
# 1. "Product Y launch date"
# 2. "Company X Product Y 2025 announcement"
# 3. "Company X new products 2025"

# All queries executed
# Results deduplicated by URL
# Top 3-4 unique sources scraped
# More targeted evidence retrieved
```

## Advanced Usage: Understanding Query Generation

```python
from src.factchecker.simple.signatures.query_generator import QueryGenerator
import dspy

# Initialize query generator
query_gen = dspy.ChainOfThought(QueryGenerator)

# Generate queries for a statement
statement = "Tesla delivered 500,000 vehicles in Q4 2023"
result = query_gen(statement=statement)

# View generated queries
print("Generated queries:")
for i, query in enumerate(result.queries, 1):
    print(f"{i}. {query}")

# Example output:
# 1. Tesla Q4 2023 deliveries
# 2. Tesla 500000 vehicles 2023
# 3. Tesla quarterly delivery numbers 2023
```

## Monitoring and Debugging

### Enable Detailed Logging

```python
import logging

logging.basicConfig(level=logging.INFO)

judge = JudgeModule()
result = judge(statement="Your statement here")

# Look for these log messages:
# - "Query generation"
# - "Search execution"
# - "Evidence gathering complete"
```

### Understanding Research Triggered

```python
result = judge(statement="Your statement")

if result.research_triggered:
    print("Web research was used (multi-query search executed)")
    # More reliable verdict with evidence
else:
    print("LLM knowledge only (no web research needed)")
    # Fast response, high confidence from LLM knowledge
```

## Best Practices

### 1. Statement Clarity
```python
# ✅ Good: Specific, verifiable claims
"Apple released the iPhone 15 in September 2023"
"The Eiffel Tower is 330 meters tall"

# ❌ Poor: Vague or subjective
"Apple makes good phones"
"Paris is beautiful"
```

### 2. Temporal Claims
```python
# ✅ Multi-query enhancement excels at:
"Company X has been operating in Market Y for 5 years"
"Product Z was discontinued in 2024"
"Event A happened before Event B"

# These benefit most from targeted temporal queries
```

### 3. Numeric Claims
```python
# ✅ Multi-query enhancement helps verify:
"Company X has 10,000 employees"
"Product Y costs $499"
"Service Z has 1 million users"

# Focused queries find specific numeric evidence
```

### 4. Compound Statements
```python
# ⚠️ Consider breaking into parts:
long_statement = "Apple released iPhone 15 in Sept 2023 and it costs $799 and has 5 cameras"

# Better:
statements = [
    "Apple released iPhone 15 in September 2023",
    "iPhone 15 costs $799",
    "iPhone 15 has 5 cameras"
]

for stmt in statements:
    result = judge(statement=stmt)
    print(f"{stmt}: {result.overall_verdict}")
```

## Performance Tips

### 1. Batch Processing
```python
statements = [...]

# Process in batch
results = []
for statement in statements:
    result = judge(statement=statement)
    results.append(result)

    # Optional: Add delay to avoid rate limits
    # time.sleep(0.5)
```

### 2. Caching Results
```python
import functools

@functools.lru_cache(maxsize=100)
def cached_judge(statement: str):
    judge = JudgeModule()
    return judge(statement=statement)

# Reuse results for duplicate statements
result = cached_judge("Your statement")
```

### 3. Filtering by Confidence
```python
result = judge(statement="Your statement")

if result.confidence >= 0.8:
    print("High confidence verdict - likely reliable")
elif result.confidence >= 0.5:
    print("Medium confidence - may need review")
else:
    print("Low confidence - manual verification recommended")
```

## Troubleshooting

### Issue: "Evidence gathering failed"

**Cause:** API service (Serper or Firecrawl) unavailable

**Solution:**
```python
# Module handles this gracefully
# Returns first pass result with note
# Check API credentials and status
```

### Issue: Slow response times

**Cause:** Multiple API calls (query generation + search + scraping)

**Solutions:**
1. Accept 7-10s latency for accuracy benefit
2. Use caching for repeated statements
3. Consider async processing for batch jobs

### Issue: Unexpected verdicts

**Debug:**
```python
result = judge(statement="Your statement")

# Examine reasoning
print("Reasoning:", result.reasoning)

# Check if research was triggered
print("Research used:", result.research_triggered)

# Review confidence score
print("Confidence:", result.confidence)
```

## Example: Complete Workflow

```python
from src.factchecker.simple.modules.judge_module import JudgeModule

def fact_check(statement: str) -> dict:
    """Fact check a statement and return structured results."""
    judge = JudgeModule()
    result = judge(statement=statement)

    return {
        'statement': statement,
        'verdict': result.overall_verdict,
        'confidence': result.confidence,
        'reasoning': result.reasoning,
        'used_web_search': result.research_triggered,
        'is_supported': result.overall_verdict == 'SUPPORTED',
        'is_refuted': result.overall_verdict == 'CONTAINS_REFUTED_CLAIMS',
        'needs_review': result.confidence < 0.5
    }

# Usage
statements = [
    "The Earth orbits the Sun",
    "Apple released iPhone 15 in September 2023",
    "Mondelez has been selling sugar-free Oreos for decades"
]

for stmt in statements:
    result = fact_check(stmt)
    print(f"\nStatement: {result['statement']}")
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Web search used: {result['used_web_search']}")

    if result['is_refuted']:
        print("⚠️  WARNING: Statement contains false information")
    elif result['needs_review']:
        print("⚠️  LOW CONFIDENCE: Manual review recommended")
    elif result['is_supported']:
        print("✓ Verified as accurate")
```

## API Reference

### JudgeModule

#### `__init__()`
Initialize the judge module with all components.

#### `forward(statement: str) -> dspy.Prediction`
Evaluate a statement for factual correctness.

**Parameters:**
- `statement` (str): The statement to evaluate

**Returns:**
- `dspy.Prediction` with fields:
  - `statement`: The input statement
  - `overall_verdict`: The factual correctness verdict
  - `confidence`: Confidence score (0.0-1.0)
  - `reasoning`: Explanation of the verdict
  - `research_triggered`: Boolean indicating if web research was used

#### `_should_trigger_research(verdict: str, confidence: float, reasoning: str) -> bool`
Determine if web research should be triggered.

#### `_gather_evidence(statement: str) -> str`
Perform multi-query web research and gather evidence.

### QueryGenerator

#### Input Fields:
- `statement` (str): The statement to generate queries for

#### Output Fields:
- `queries` (list[str]): 1-3 focused search queries

## Migration from Old Version

No code changes needed! The enhancement is backward compatible.

**Old code still works:**
```python
# This continues to work exactly as before
judge = JudgeModule()
result = judge(statement="Your statement")
```

**What changed internally:**
- More targeted search queries
- More sources scraped (3-4 instead of 2)
- Better deduplication
- Higher accuracy for temporal/numeric claims

## Resources

- **Implementation Details:** See `MULTI_QUERY_ENHANCEMENT_SUMMARY.md`
- **Flow Diagrams:** See `MULTI_QUERY_FLOW_DIAGRAM.md`
- **Technical Notes:** See `IMPLEMENTATION_NOTES.md`
- **Test Script:** Run `python test_multi_query_enhancement.py`
