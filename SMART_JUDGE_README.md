# SmartJudgeModule - Intelligent Fact-Checking Router

## Overview

`SmartJudgeModule` is an intelligent routing layer that automatically delegates fact-checking requests to the most appropriate strategy based on the input characteristics. It serves as the new primary entry point for fact-checking operations, replacing direct usage of `JudgeModule` in evaluation contexts.

## Architecture

The module wraps two core fact-checking components:
1. **JudgeModule** - Fast, LLM-only evaluation without web research
2. **FactCheckerPipeline** - Full pipeline with web research, claim extraction, and iterative verification

## Routing Logic

The module implements a sophisticated decision tree:

### Route 1: URL-Based Routing
**Trigger**: URLs provided in the `urls` parameter
**Action**: Pre-seeds `FactCheckerPipeline` with scraped evidence from provided URLs
**Use Case**: When you have specific sources you want the fact-checker to consider

```python
smart_judge = SmartJudgeModule()
result = smart_judge(
    statement="The capital of France is Paris",
    urls=["https://en.wikipedia.org/wiki/Paris"]
)
```

### Route 2: Temporal Detection
**Trigger**: Statement contains dates/events after June 2024 or future references
**Action**: Routes directly to `FactCheckerPipeline` for web research
**Use Case**: Recent events, current statistics, or future predictions

Examples of temporal claims:
- "In 2024, the GDP growth rate was..."
- "The current president of..."
- "This year's Nobel Prize winner is..."
- "Recent studies show that..."

```python
result = smart_judge(
    statement="In 2025, the global GDP growth rate exceeded 4%"
)
# Automatically routes to web research
```

### Route 3: Confidence-Based Fallback
**Trigger**: No URLs or temporal claims detected
**Action**:
1. First tries `JudgeModule` (fast LLM-only evaluation)
2. If confidence < 0.6 OR verdict is `CONTAINS_UNSUPPORTED_CLAIMS`, falls back to `FactCheckerPipeline`
3. Otherwise, returns the high-confidence `JudgeModule` result

```python
# High confidence historical fact -> uses JudgeModule
result = smart_judge(
    statement="World War II ended in 1945"
)

# Low confidence or uncertain claim -> falls back to web research
result = smart_judge(
    statement="The Eiffel Tower was built using exactly 18,038 pieces of iron"
)
```

## Implementation Details

### URL Pre-Seeding

When URLs are provided, the module:
1. Uses `FirecrawlService` to scrape each URL
2. Formats the scraped content as structured evidence
3. Passes this evidence as `initial_evidence` to `FactCheckerPipeline`
4. The pipeline prepends this evidence to the FireJudge iterations

The formatted evidence structure:
```
--- Pre-seeded Evidence from https://example.com ---
Title: Example Page Title
Content: [Full markdown content]
```

### Temporal Detection

Uses a lightweight DSPy `ChainOfThought` with the `TemporalDetector` signature to analyze statements for:
- Year references >= 2024
- Future dates (month/year combinations indicating future)
- Temporal indicators ("recently", "this year", "currently")
- Status claims that change over time ("current president", "latest version")

### Modified Pipeline Components

**FireJudgeModule** (`fire_judge_module.py`):
- Added `initial_evidence: str = ""` parameter to `forward()`
- Pre-existing evidence is used before any web searches

**FactCheckerPipeline** (`fact_checker_pipeline.py`):
- Added `initial_evidence: str = ""` parameter to `forward()`
- Passes initial evidence to all claim evaluations via `FireJudgeModule`

## API Reference

### SmartJudgeModule

```python
class SmartJudgeModule(dspy.Module):
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        max_judge_iterations: int = 3,
        max_page_visits: int = 3
    )
```

**Parameters:**
- `confidence_threshold` (float): Minimum confidence to trust JudgeModule verdict without fallback (default: 0.6)
- `max_judge_iterations` (int): Max iterations for FactCheckerPipeline (default: 3)
- `max_page_visits` (int): Max pages to visit per search query (default: 3)

### forward()

```python
def forward(
    self,
    statement: str,
    urls: Optional[list[str]] = None
) -> dspy.Prediction
```

**Parameters:**
- `statement` (str): The statement to fact-check
- `urls` (Optional[list[str]]): Optional URLs to use as evidence sources

**Returns:**
`dspy.Prediction` with:
- `statement` (str): The input statement
- `overall_verdict` (str): "SUPPORTED" | "CONTAINS_UNSUPPORTED_CLAIMS" | "CONTAINS_REFUTED_CLAIMS"
- `confidence` (float): Confidence score between 0.0 and 1.0
- `reasoning` (str): Explanation of the verdict
- `routing_decision` (str): Description of which routing path was taken
- `claims` (list[str]): Extracted claims (only present if pipeline was used)
- `claim_results` (list): Claim-level results (only present if pipeline was used)

## Usage Examples

### Basic Usage

```python
from src.factchecker.modules import SmartJudgeModule

smart_judge = SmartJudgeModule()

# Simple fact-checking
result = smart_judge(statement="The Earth orbits the Sun")
print(f"Verdict: {result.overall_verdict}")
print(f"Confidence: {result.confidence}")
print(f"Route taken: {result.routing_decision}")
```

### With Custom Confidence Threshold

```python
# More aggressive web research (lower threshold)
smart_judge = SmartJudgeModule(confidence_threshold=0.8)

result = smart_judge(statement="Python was created by Guido van Rossum")
# Will likely fall back to web research with higher threshold
```

### With Pre-Seeded URLs

```python
result = smart_judge(
    statement="OpenAI released GPT-4 in March 2023",
    urls=[
        "https://openai.com/research/gpt-4",
        "https://en.wikipedia.org/wiki/GPT-4"
    ]
)
# Uses provided URLs as initial evidence
```

### Temporal Claims

```python
# Automatically detects temporal nature
result = smart_judge(
    statement="In 2024, renewable energy accounted for 30% of global electricity"
)
print(result.routing_decision)
# Output: "Temporal claim detected (recent/future dates) - routing to FactCheckerPipeline for web research"
```

## Integration with Existing Code

Replace existing `JudgeModule` usage in evaluation contexts:

**Before:**
```python
from src.factchecker.simple.modules import JudgeModule

judge = JudgeModule()
result = judge(statement=statement)
```

**After:**
```python
from src.factchecker.modules import SmartJudgeModule

smart_judge = SmartJudgeModule()
result = smart_judge(statement=statement)
# Automatically routes to the best strategy
```

## Performance Considerations

### Cost Optimization
- JudgeModule path: ~1 LLM call (cheapest, fastest)
- Temporal detection: +1 lightweight LLM call
- Full pipeline: Multiple LLM calls for claim extraction, page selection, evidence summarization, and iterative judgment

### Latency
- JudgeModule path: ~1-3 seconds
- Pipeline path: ~30-60 seconds (depends on web searches)

### Tuning Recommendations
- **Lower `confidence_threshold`** (e.g., 0.5): More aggressive, uses JudgeModule more often
- **Higher `confidence_threshold`** (e.g., 0.8): More cautious, falls back to web research more often
- **Reduce `max_page_visits`**: Faster but less thorough research
- **Reduce `max_judge_iterations`**: Fewer research iterations per claim

## Testing

Run the test script:

```bash
python test_smart_judge.py
```

Expected output shows routing decisions for:
1. Simple fact (JudgeModule route)
2. Temporal claim (Pipeline route)
3. URL-provided statement (Pipeline with pre-seeding)

## Files Modified/Created

### New Files
- `src/factchecker/modules/smart_judge_module.py` - Main routing module
- `src/factchecker/signatures/temporal_detector.py` - Temporal claim detection signature
- `test_smart_judge.py` - Test script
- `SMART_JUDGE_README.md` - This documentation

### Modified Files
- `src/factchecker/modules/fire_judge_module.py` - Added `initial_evidence` parameter
- `src/factchecker/modules/fact_checker_pipeline.py` - Added `initial_evidence` parameter
- `src/factchecker/modules/__init__.py` - Added `SmartJudgeModule` export
- `src/factchecker/signatures/__init__.py` - Added `TemporalDetector` export

## Future Enhancements

Potential improvements:
1. **Caching**: Cache temporal detection results for repeated statements
2. **Hybrid routing**: Use partial web research (fewer iterations) for medium confidence
3. **Source quality scoring**: Prioritize high-authority URLs in pre-seeding
4. **Adaptive thresholds**: Learn optimal confidence thresholds from user feedback
5. **Parallel evaluation**: Run both paths simultaneously and compare results
