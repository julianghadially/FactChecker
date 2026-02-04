# Quick Start Guide - Temporal Router

## Installation

No additional dependencies required. The temporal router uses only standard Python libraries and existing project dependencies.

## Basic Usage

### 1. Import and Initialize

```python
from src.factchecker.modules.temporal_router_module import TemporalRouterModule

# Create router with default settings
router = TemporalRouterModule()
```

### 2. Fact-Check a Statement

```python
# Simple statement
result = router(statement="The Apollo 11 mission landed in 1969.")

# Access results
print(f"Verdict: {result.overall_verdict}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Route used: {result.route_decision}")
print(f"Reasoning: {result.reasoning}")
```

### 3. With Priority URLs

```python
# Provide evidence URLs
urls = [
    "https://example.com/report",
    "https://news.com/article"
]

result = router(
    statement="Company X reported $1B revenue in Q4.",
    urls=urls
)
```

## Command Line Usage

### Single Statement Check

```bash
python src/main.py --mode check \
    --statement "In 2025, AI adoption increased significantly."
```

### Batch Evaluation

```bash
python src/main.py --mode evaluate \
    --sample-size 100 \
    --dataset-path data/FactChecker_news_claims.csv
```

## Configuration

### Custom Knowledge Cutoff

```python
from datetime import datetime

router = TemporalRouterModule(
    knowledge_cutoff=datetime(2024, 9, 1)  # Sept 2024
)
```

### Adjust Research Parameters

```python
router = TemporalRouterModule(
    max_judge_iterations=5,  # More research iterations
    max_page_visits=5        # Visit more pages per search
)
```

## Understanding Results

### Result Object Structure

```python
result = router(statement="...")

# Common fields (both routes)
result.statement          # Input statement
result.overall_verdict    # SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
result.confidence         # 0.0 to 1.0
result.reasoning          # Explanation
result.route_decision     # "judge" or "pipeline"
result.route_reason       # Why this route was chosen

# Additional fields (pipeline route only)
result.claims             # List of extracted claims
result.claim_results      # Detailed results per claim
```

### Verdict Types

- `SUPPORTED`: Statement is factually correct
- `CONTAINS_UNSUPPORTED_CLAIMS`: Some claims lack evidence
- `CONTAINS_REFUTED_CLAIMS`: Some claims are false

## Routing Behavior

### Fast Path (JudgeModule)

**Used when:**
- No URLs provided
- All dates before June 2024
- No temporal keywords (today, recent, latest, etc.)

**Characteristics:**
- ⚡ Fast: 1-3 seconds
- 💰 Cheap: 1 API call
- 📚 Uses LLM knowledge only

**Example:**
```python
router(statement="World War II ended in 1945.")
# → JudgeModule (historical fact)
```

### Web Research Path (FactCheckerPipeline)

**Used when:**
- URLs provided (in statement or parameter)
- Dates ≥ June 2024
- Temporal keywords present

**Characteristics:**
- 🔍 Thorough: 10-30 seconds
- 💵 More expensive: 15-30 API calls
- 🌐 Uses current web data

**Example:**
```python
router(statement="In January 2025, tech layoffs increased.")
# → FactCheckerPipeline (recent event)
```

## Common Patterns

### Pattern 1: Historical Facts

```python
# Historical statements → Fast path
statements = [
    "The moon landing occurred in 1969.",
    "Einstein published relativity in 1905.",
    "The Berlin Wall fell in 1989."
]

for stmt in statements:
    result = router(statement=stmt)
    assert result.route_decision == "judge"  # Fast path
```

### Pattern 2: Recent Events

```python
# Recent statements → Web research
statements = [
    "In 2025, the economy grew.",
    "Today's weather forecast predicts rain.",
    "Recent studies show climate change."
]

for stmt in statements:
    result = router(statement=stmt)
    assert result.route_decision == "pipeline"  # Web research
```

### Pattern 3: With Evidence URLs

```python
# URLs provided → Web research (with priority)
result = router(
    statement="The product launched successfully.",
    urls=["https://company.com/press-release"]
)
assert result.route_decision == "pipeline"
```

## Testing the Router

### Run Unit Tests

```bash
# Run all tests
python -m unittest tests.test_temporal_router -v

# Should see: Ran 27 tests in ~0.1s OK
```

### Run Demo Script

```bash
# Interactive demo
python examples/temporal_router_demo.py
```

## Debugging

### Enable Detailed Logging

The router automatically logs routing decisions:

```
============================================================
TEMPORAL ROUTING DECISION
============================================================
Statement: The Apollo 11 mission landed...
URLs found: 0
Dates found: 1
  - 1969-07-20
Route: JudgeModule (fast evaluation)
Reason: No temporal references or URLs requiring web research
============================================================
```

### Check Routing Logic

```python
router = TemporalRouterModule()

# Test date extraction
dates = router._extract_dates("In January 2025")
print(dates)  # [datetime(2025, 1, 1)]

# Test URL extraction
urls = router._extract_urls("Visit https://example.com")
print(urls)  # ['https://example.com']

# Test temporal keywords
has_keywords = router._has_temporal_keywords("The latest report")
print(has_keywords)  # True

# Test routing decision
should_use_web, reason = router._should_use_web_research(
    statement="Today's news",
    urls=[],
    dates=[]
)
print(f"{should_use_web}: {reason}")
# True: Temporal keywords suggest recent/current events
```

## Performance Tips

### 1. Batch Processing

```python
# Process multiple statements
statements = [...]  # Your statements

results = []
for stmt in statements:
    result = router(statement=stmt)
    results.append(result)
```

### 2. Cache Results

```python
# Cache verdicts for repeated statements
cache = {}

def cached_check(statement: str):
    if statement not in cache:
        cache[statement] = router(statement=statement)
    return cache[statement]
```

### 3. Parallelize (Careful!)

```python
from concurrent.futures import ThreadPoolExecutor

def check_statement(stmt: str):
    return router(statement=stmt)

with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(check_statement, statements))
```

## Troubleshooting

### Issue: All statements go to pipeline

**Cause**: Knowledge cutoff too far in past

**Fix**:
```python
from datetime import datetime
router = TemporalRouterModule(
    knowledge_cutoff=datetime(2024, 6, 1)
)
```

### Issue: Priority URLs not used

**Cause**: Not passing URLs parameter

**Fix**:
```python
# Correct
result = router(statement="...", urls=["https://..."])

# Incorrect
result = router(statement="...")  # URLs in statement text only
```

### Issue: Wrong routing decision

**Cause**: Statement contains unexpected date/keyword

**Debug**:
```python
router = TemporalRouterModule()
dates = router._extract_dates(statement)
keywords = router._has_temporal_keywords(statement)
print(f"Dates: {dates}, Keywords: {keywords}")
```

## Examples

### Example 1: Check Multiple Statements

```python
from src.factchecker.modules.temporal_router_module import TemporalRouterModule

router = TemporalRouterModule()

statements = [
    "The Eiffel Tower was completed in 1889.",
    "In 2025, renewable energy usage increased.",
    "Recent studies show benefits of meditation."
]

for stmt in statements:
    result = router(statement=stmt)
    print(f"\nStatement: {stmt}")
    print(f"Route: {result.route_decision}")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence:.0%}")
```

### Example 2: With Custom Configuration

```python
from datetime import datetime

# More aggressive research settings
router = TemporalRouterModule(
    max_judge_iterations=5,
    max_page_visits=5,
    knowledge_cutoff=datetime(2023, 1, 1)
)

result = router(statement="Economic trends in 2024")
```

### Example 3: Priority URL Workflow

```python
# 1. User provides statement + evidence URLs
statement = "Company X acquired Company Y for $10B."
urls = [
    "https://company-x.com/press-release",
    "https://techcrunch.com/acquisition"
]

# 2. Router processes with priority
result = router(statement=statement, urls=urls)

# 3. Priority URLs scraped first, then web search if needed
print(f"Route: {result.route_decision}")  # "pipeline"
print(f"Reason: {result.route_reason}")   # "URLs provided (2 URLs found)"
```

## Best Practices

### ✅ Do

- Use priority URLs when you have relevant sources
- Let the router decide the route automatically
- Check `route_decision` to understand system behavior
- Test with your specific domain statements
- Monitor cost/latency for your use case

### ❌ Don't

- Hardcode routing decisions (defeats purpose)
- Ignore routing reason (useful for debugging)
- Assume all historical facts → judge (router is smart)
- Use without testing on your data first
- Forget to handle errors gracefully

## Integration Examples

### Flask API

```python
from flask import Flask, request, jsonify
from src.factchecker.modules.temporal_router_module import TemporalRouterModule

app = Flask(__name__)
router = TemporalRouterModule()

@app.route('/fact-check', methods=['POST'])
def fact_check():
    data = request.json
    statement = data.get('statement')
    urls = data.get('urls', [])

    result = router(statement=statement, urls=urls)

    return jsonify({
        'verdict': result.overall_verdict,
        'confidence': result.confidence,
        'reasoning': result.reasoning,
        'route': result.route_decision,
        'route_reason': result.route_reason
    })
```

### CLI Tool

```python
import argparse
from src.factchecker.modules.temporal_router_module import TemporalRouterModule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('statement', help='Statement to check')
    parser.add_argument('--urls', nargs='+', help='Evidence URLs')
    args = parser.parse_args()

    router = TemporalRouterModule()
    result = router(statement=args.statement, urls=args.urls)

    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Route: {result.route_decision} ({result.route_reason})")

if __name__ == '__main__':
    main()
```

## Getting Help

- **Documentation**: See `docs/temporal_router.md`
- **Tests**: Check `tests/test_temporal_router.py` for examples
- **Demo**: Run `examples/temporal_router_demo.py`
- **Summary**: Read `TEMPORAL_ROUTER_SUMMARY.md`

## Next Steps

1. Read the full documentation: `docs/temporal_router.md`
2. Run the demo: `python examples/temporal_router_demo.py`
3. Run tests: `python -m unittest tests.test_temporal_router -v`
4. Try your own statements
5. Monitor routing decisions for your use case
6. Adjust configuration as needed

---

**Ready to start?** Just import and use:

```python
from src.factchecker.modules.temporal_router_module import TemporalRouterModule

router = TemporalRouterModule()
result = router(statement="Your statement here")

print(f"Verdict: {result.overall_verdict}")
print(f"Route: {result.route_decision}")
```

Happy fact-checking! 🔍✅
