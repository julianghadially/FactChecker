# AdaptiveJudgeModule

## Overview

`AdaptiveJudgeModule` is an intelligent fact-checking module that combines the speed of `JudgeModule` with the thoroughness of `FactCheckerPipeline`. It uses the model's confidence score as a natural decision boundary to determine when external research is needed.

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    AdaptiveJudgeModule                      │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
                  ┌──────────────────┐
                  │  1. JudgeModule  │
                  │  (Fast, no web)  │
                  └──────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │ Check verdict & confidence│
              └──────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
    ┌─────────────────────┐   ┌──────────────────────┐
    │ High confidence OR  │   │ UNSUPPORTED claims   │
    │ Other verdict types │   │ + Low confidence     │
    └─────────────────────┘   └──────────────────────┘
                │                         │
                │                         ▼
                │         ┌──────────────────────────┐
                │         │ 2. FactCheckerPipeline   │
                │         │ (Thorough, with web)     │
                │         └──────────────────────────┘
                │                         │
                └────────────┬────────────┘
                             ▼
                      Return result
```

## Key Features

1. **Intelligent Routing**: Automatically decides when web research is needed
2. **Confidence-Based**: Uses the model's own uncertainty signal
3. **Lazy Initialization**: Only loads FactCheckerPipeline when needed
4. **Configurable**: Adjustable confidence threshold and fallback behavior
5. **Transparent**: Returns `fallback_triggered` flag for observability

## When Fallback is Triggered

Fallback to `FactCheckerPipeline` occurs when **ALL** of these conditions are met:

1. ✅ `enable_fallback=True` (default)
2. ✅ Verdict is `"CONTAINS_UNSUPPORTED_CLAIMS"`
3. ✅ Confidence < `confidence_threshold` (default: 0.7)

## Usage

### Basic Usage

```python
import dspy
from src.factchecker.modules.adaptive_judge_module import AdaptiveJudgeModule

# Configure DSPy
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

# Initialize with defaults
adaptive_judge = AdaptiveJudgeModule()

# Evaluate a statement
result = adaptive_judge(statement="The Earth orbits the Sun.")

print(f"Verdict: {result.overall_verdict}")
print(f"Confidence: {result.confidence}")
print(f"Fallback triggered: {result.fallback_triggered}")
print(f"Reasoning: {result.reasoning}")
```

### Custom Configuration

```python
# More aggressive fallback (lower threshold)
adaptive_judge = AdaptiveJudgeModule(
    confidence_threshold=0.5,  # Trigger fallback more easily
    enable_fallback=True,
    max_judge_iterations=3,    # For pipeline
    max_page_visits=3          # For pipeline
)

# Conservative mode (higher threshold)
adaptive_judge = AdaptiveJudgeModule(
    confidence_threshold=0.9,  # Rarely trigger fallback
    enable_fallback=True
)

# Judge-only mode (no fallback)
adaptive_judge = AdaptiveJudgeModule(
    enable_fallback=False  # Always use JudgeModule
)
```

## Parameters

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confidence_threshold` | float | 0.7 | Minimum confidence to accept `CONTAINS_UNSUPPORTED_CLAIMS` without fallback (0.0-1.0) |
| `enable_fallback` | bool | True | Whether to enable automatic fallback to pipeline |
| `max_judge_iterations` | int | 3 | Max search iterations per claim in pipeline |
| `max_page_visits` | int | 3 | Max pages to visit per search query in pipeline |

### Return Values

The `forward()` method returns a `dspy.Prediction` with:

| Field | Type | Description |
|-------|------|-------------|
| `statement` | str | The input statement |
| `overall_verdict` | str | One of: `SUPPORTED`, `CONTAINS_UNSUPPORTED_CLAIMS`, `CONTAINS_REFUTED_CLAIMS` |
| `confidence` | float | Confidence score (0.0-1.0) |
| `reasoning` | str | Explanation of the verdict |
| `fallback_triggered` | bool | Whether pipeline fallback was used |
| `claims` | list[str] | Claims extracted (only if fallback triggered) |
| `claim_results` | list | Detailed claim results (only if fallback triggered) |

## Example Scenarios

### Scenario 1: Well-Known Fact (No Fallback)

```python
result = adaptive_judge(statement="Water boils at 100°C at sea level.")

# Expected:
# - overall_verdict: "SUPPORTED"
# - confidence: ~0.95
# - fallback_triggered: False
# - Fast response (~1-2 seconds)
```

### Scenario 2: Obviously False (No Fallback)

```python
result = adaptive_judge(statement="The Moon is made of cheese.")

# Expected:
# - overall_verdict: "CONTAINS_REFUTED_CLAIMS"
# - confidence: ~0.98
# - fallback_triggered: False
# - Fast response (~1-2 seconds)
```

### Scenario 3: Obscure/Recent Fact (Fallback Triggered)

```python
result = adaptive_judge(
    statement="Company X's Q4 2024 revenue was $523 million."
)

# Expected:
# - Initial judge: "CONTAINS_UNSUPPORTED_CLAIMS", confidence: ~0.4
# - Fallback triggered: True
# - Pipeline performs web research
# - Final verdict based on found evidence
# - Slower response (~10-30 seconds)
```

### Scenario 4: Uncertain Domain-Specific Claim (Fallback Triggered)

```python
result = adaptive_judge(
    statement="The new Tesla Model Z has a 600-mile range."
)

# Expected:
# - Initial judge uncertain about recent product specs
# - Fallback triggered: True
# - Pipeline searches for current information
# - Returns evidence-based verdict
```

## Best Practices

### 1. Choose the Right Confidence Threshold

```python
# For critical applications (medical, legal, financial)
# Use higher threshold to trigger fallback more often
adaptive_judge = AdaptiveJudgeModule(confidence_threshold=0.8)

# For general use cases
# Use default threshold
adaptive_judge = AdaptiveJudgeModule(confidence_threshold=0.7)

# For low-stakes applications or when speed is critical
# Use lower threshold or disable fallback
adaptive_judge = AdaptiveJudgeModule(confidence_threshold=0.5)
```

### 2. Monitor Fallback Rate

```python
fallback_count = 0
total_count = 0

for statement in statements:
    result = adaptive_judge(statement=statement)
    total_count += 1
    if result.fallback_triggered:
        fallback_count += 1

fallback_rate = fallback_count / total_count
print(f"Fallback rate: {fallback_rate:.1%}")

# Adjust threshold based on observed rate:
# - Too high (>50%): Consider lowering threshold or using pipeline directly
# - Too low (<5%): Threshold might be too permissive
```

### 3. Log Fallback Events

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

result = adaptive_judge(statement=statement)

if result.fallback_triggered:
    logger.warning(
        f"Fallback triggered for statement: {statement[:100]}... "
        f"Initial confidence: {result.confidence}"
    )
```

### 4. Handle Pipeline Failures

```python
try:
    result = adaptive_judge(statement=statement)
except Exception as e:
    logger.error(f"Error during fact-checking: {e}")
    # Fallback to judge-only mode or return error
    fallback_judge = AdaptiveJudgeModule(enable_fallback=False)
    result = fallback_judge(statement=statement)
```

## Performance Characteristics

| Scenario | Speed | Accuracy | Cost |
|----------|-------|----------|------|
| **High-confidence judge result** | ⚡ Fast (1-2s) | ✅ Good for known facts | 💰 Low (1 LLM call) |
| **Low-confidence + fallback** | 🐢 Slower (10-30s) | ✅✅ High (web research) | 💰💰💰 Higher (multiple calls + API) |
| **Judge-only mode** | ⚡ Fast (1-2s) | ⚠️ Limited to LLM knowledge | 💰 Low (1 LLM call) |

## API Key Requirements

### Required Always
- `OPENAI_API_KEY` (or your configured LLM provider)

### Required Only for Fallback
- `SERPER_API_KEY` (for web search)
- `FIRECRAWL_API_KEY` (for page scraping)

**Note**: If fallback is disabled, only the LLM API key is needed.

## Debugging

Enable debug logging to see decision flow:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

adaptive_judge = AdaptiveJudgeModule()
result = adaptive_judge(statement="Some statement")
```

This will show:
- Initial JudgeModule verdict and confidence
- Fallback decision logic
- Pipeline initialization (if triggered)
- Final verdict

## Comparison with Other Modules

| Module | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| **JudgeModule** | ⚡⚡⚡ Very Fast | ✅ Good for known facts | Quick checks, high-volume |
| **FactCheckerPipeline** | 🐢 Slow | ✅✅✅ Very High | Critical verification, research |
| **AdaptiveJudgeModule** | ⚡/🐢 Adaptive | ✅✅ High | General purpose, balanced |

## Advanced: Batch Processing

```python
def batch_fact_check(statements, confidence_threshold=0.7):
    """Efficiently fact-check multiple statements."""
    adaptive_judge = AdaptiveJudgeModule(
        confidence_threshold=confidence_threshold
    )

    results = []
    for statement in statements:
        result = adaptive_judge(statement=statement)
        results.append({
            'statement': statement,
            'verdict': result.overall_verdict,
            'confidence': result.confidence,
            'fallback_used': result.fallback_triggered
        })

    return results

# Usage
statements = [
    "The Earth is round.",
    "Recent studies show X causes Y.",
    "Company Z announced new product."
]

results = batch_fact_check(statements, confidence_threshold=0.7)

for r in results:
    print(f"{r['verdict']:30} (conf: {r['confidence']:.2f}) - {r['statement'][:50]}...")
```

## Limitations

1. **Confidence Calibration**: LLM confidence scores may not always perfectly reflect uncertainty
2. **Latency Variability**: Response time varies greatly depending on fallback trigger
3. **Cost Unpredictability**: Fallback triggering affects cost per request
4. **Binary Decision**: No partial fallback - either full pipeline or none

## Future Enhancements

Potential improvements:
- [ ] Adaptive threshold based on statement type
- [ ] Partial fallback (limited web search)
- [ ] Confidence calibration training
- [ ] Async/parallel processing support
- [ ] Caching of recent verdicts
- [ ] Custom fallback strategies

## Contributing

To add features or fix bugs:
1. Add tests in `tests/test_adaptive_judge_module.py`
2. Update documentation
3. Submit PR with clear description

## License

Same as parent project.
