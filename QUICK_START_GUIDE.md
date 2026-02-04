# SmartJudgeModule Quick Start Guide

## Installation
No additional installation needed - the module uses existing dependencies.

## Basic Usage

```python
from src.factchecker.modules import SmartJudgeModule

# Initialize
smart_judge = SmartJudgeModule()

# Fact-check a statement
result = smart_judge(statement="Your statement here")

# Access results
print(f"Verdict: {result.overall_verdict}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")
print(f"Route taken: {result.routing_decision}")
```

## Common Scenarios

### 1. Simple Fact-Checking (Fastest)
```python
result = smart_judge(statement="The Earth orbits the Sun")
# Uses JudgeModule (fast, 1-3 seconds)
```

### 2. Recent Events (Automatic Web Research)
```python
result = smart_judge(statement="In 2024, inflation rates in the US were...")
# Automatically detects temporal claim and uses web research
```

### 3. With Specific Sources
```python
result = smart_judge(
    statement="Climate change affects ocean temperatures",
    urls=["https://climate.nasa.gov/", "https://www.ipcc.ch/"]
)
# Pre-seeds fact-checker with your provided sources
```

### 4. More Thorough Checking (Lower Threshold)
```python
smart_judge = SmartJudgeModule(confidence_threshold=0.4)
result = smart_judge(statement="...")
# More likely to use fast JudgeModule path
```

### 5. More Cautious Checking (Higher Threshold)
```python
smart_judge = SmartJudgeModule(confidence_threshold=0.8)
result = smart_judge(statement="...")
# More likely to fall back to web research
```

## Understanding Verdicts

- **SUPPORTED**: Statement is factually correct
- **CONTAINS_UNSUPPORTED_CLAIMS**: Cannot verify (insufficient evidence)
- **CONTAINS_REFUTED_CLAIMS**: Statement contains false information

## Understanding Routing Decisions

The `routing_decision` field explains which path was taken:

```python
# Example outputs:
"No URLs or temporal claims - trying JudgeModule first -> High confidence (0.92) - using JudgeModule result"
"Temporal claim detected (recent/future dates) - routing to FactCheckerPipeline for web research"
"URLs provided (2 URLs) - routing to FactCheckerPipeline with pre-seeded evidence"
"No URLs or temporal claims - trying JudgeModule first -> Falling back to FactCheckerPipeline (low confidence (0.45 < 0.6))"
```

## Configuration Options

```python
SmartJudgeModule(
    confidence_threshold=0.6,    # Min confidence to trust JudgeModule (0.0-1.0)
    max_judge_iterations=3,      # Max research iterations per claim
    max_page_visits=3            # Max web pages to visit per search
)
```

## When to Use What

| Scenario | Recommended Setting |
|----------|-------------------|
| Fast screening | `confidence_threshold=0.4` |
| Balanced (default) | `confidence_threshold=0.6` |
| High accuracy needed | `confidence_threshold=0.8` |
| Have specific sources | Pass `urls` parameter |
| Recent events | Default (auto-detects) |
| Historical facts | Default (uses fast path) |

## Performance Guide

### Speed
- **Fast path** (JudgeModule): 1-3 seconds
- **Web research path** (Pipeline): 30-60 seconds

### Cost (approximate)
- **Fast path**: ~$0.001 per query
- **Web research path**: ~$0.05-0.15 per query

### Quality
- **Fast path**: Good for well-known facts
- **Web research path**: Better for:
  - Obscure facts
  - Recent events
  - Controversial claims
  - When you have specific sources

## Complete Example

```python
from src.factchecker.modules import SmartJudgeModule

# Initialize with custom settings
smart_judge = SmartJudgeModule(
    confidence_threshold=0.7,
    max_page_visits=2
)

# Fact-check with sources
result = smart_judge(
    statement="The latest IPCC report states that global temperatures have risen 1.1°C since pre-industrial times",
    urls=["https://www.ipcc.ch/report/ar6/"]
)

# Display results
print(f"\nStatement: {result.statement}")
print(f"Verdict: {result.overall_verdict}")
print(f"Confidence: {result.confidence:.2%}")
print(f"\nReasoning:\n{result.reasoning}")
print(f"\nRoute taken:\n{result.routing_decision}")

# Check if detailed results available
if hasattr(result, 'claims'):
    print(f"\nClaims extracted: {len(result.claims)}")
    for i, claim in enumerate(result.claims, 1):
        print(f"  {i}. {claim}")
```

## Troubleshooting

### Issue: Always using slow path
**Solution**: Lower `confidence_threshold` (e.g., 0.4)

### Issue: Want more thorough checking
**Solution**: Raise `confidence_threshold` (e.g., 0.8) or provide URLs

### Issue: Need faster results
**Solutions**:
- Lower `confidence_threshold` for more fast-path usage
- Reduce `max_page_visits` (e.g., 1-2)
- Reduce `max_judge_iterations` (e.g., 2)

### Issue: Getting "CONTAINS_UNSUPPORTED_CLAIMS" too often
**Solution**: This triggers automatic fallback to web research - it's working as intended!

## Migration from JudgeModule

### Old Code
```python
from src.factchecker.simple.modules import JudgeModule
judge = JudgeModule()
result = judge(statement=statement)
```

### New Code (Drop-in Replacement)
```python
from src.factchecker.modules import SmartJudgeModule
smart_judge = SmartJudgeModule()
result = smart_judge(statement=statement)
```

Same return signature, but with intelligent routing!

## Next Steps

- Read `SMART_JUDGE_README.md` for complete documentation
- Run `python example_smart_judge_usage.py` to see examples
- Run `python test_smart_judge.py` to test basic functionality
- Check `IMPLEMENTATION_SUMMARY.md` for technical details

## Support

For issues or questions:
1. Check `SMART_JUDGE_README.md` for detailed explanations
2. Review `example_smart_judge_usage.py` for usage patterns
3. Examine `routing_decision` in results to understand behavior
