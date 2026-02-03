# JudgeModule Enhancement: Web Search Integration

## Overview

The `JudgeModule` in `src/factchecker/simple/modules/judge_module.py` has been enhanced with automatic web search capability to handle recent events and verifiable facts beyond the LLM's training data. This addresses the systematic 0.5 scores (partial credit) that occur when the module returns `CONTAINS_UNSUPPORTED_CLAIMS` for statements about 2025 events it cannot verify.

## Problem Statement

The original `JudgeModule` relied solely on LLM knowledge for fact-checking. When evaluating statements about recent events (e.g., 2025 events), the LLM would indicate knowledge cutoff limitations and return:
- Verdict: `CONTAINS_UNSUPPORTED_CLAIMS`
- Confidence: ~0.5
- Reasoning: Mentions "knowledge cutoff", "training data limitations", "cannot verify", etc.

According to the `gepa_metric` function (line 56-58 in `src/optimizer/gepa_optimize.py`), these "UNKNOWN" predictions receive partial credit (0.5 score) instead of full credit (1.0) or no credit (0.0).

## Solution

The enhanced `JudgeModule` now implements a two-stage evaluation process:

### Stage 1: Initial LLM Judgment
- Calls `self.judge(statement)` to get initial verdict using LLM knowledge
- Examines the reasoning and verdict for knowledge limitation indicators

### Stage 2: Web Search Enhancement (Conditional)
If knowledge limitations are detected:
1. **Performs web search** via `SerperService.search()` for top 5 relevant results
2. **Formats search results** with titles, URLs, and snippets
3. **Re-evaluates** using `JudgeWithContext` signature with:
   - Original statement
   - Formatted search results
   - Initial reasoning (for context)
4. **Returns updated verdict** based on web evidence

## Implementation Details

### New Files Created

1. **`src/factchecker/simple/signatures/judge_with_context.py`**
   - New DSPy signature for judging with web search context
   - Input fields: `statement`, `search_results`, `initial_reasoning`
   - Output fields: `reasoning`, `verdict`, `confidence`

2. **`test_judge_with_search.py`**
   - Test script demonstrating the enhanced functionality
   - Compares behavior with and without web search enabled

### Modified Files

1. **`src/factchecker/simple/modules/judge_module.py`**
   - Added `enable_web_search` parameter (default: `True`)
   - Integrated `SerperService` for web search
   - Implemented `_detect_knowledge_limitations()` method
   - Implemented `_perform_web_search()` method
   - Enhanced `forward()` method with two-stage evaluation

2. **`src/factchecker/simple/signatures/__init__.py`**
   - Added export for `JudgeWithContext` signature

### Key Methods

#### `_detect_knowledge_limitations(reasoning: str, verdict: str) -> bool`

Detects if the LLM's reasoning indicates knowledge cutoff or uncertainty by:
- Checking if verdict is `CONTAINS_UNSUPPORTED_CLAIMS`
- Pattern matching against 20+ limitation indicators:
  - "knowledge cutoff"
  - "training data"
  - "cannot verify"
  - "beyond my knowledge"
  - "recent event"
  - "uncertain"
  - Date references (e.g., "after 2024")
  - And more...

#### `_perform_web_search(statement: str) -> str`

Performs web search and formats results:
- Uses the statement as search query (future: extract key terms with LLM)
- Retrieves top 5 results via `SerperService.search()`
- Formats each result with title, URL, and snippet
- Returns formatted string for LLM consumption
- Returns empty string on failure

### Return Value Enhancement

The `forward()` method now returns a `dspy.Prediction` with an additional field:
- `web_search_performed`: Boolean indicating whether web search was triggered

## Usage

### With Web Search (Default)

```python
import dspy
from src.factchecker.simple.modules.judge_module import JudgeModule

# Configure DSPy
dspy.configure(lm=dspy.LM("openai/gpt-5-mini", api_key=openai_key))

# Create judge with web search enabled (default)
judge = JudgeModule(enable_web_search=True)

# Evaluate a statement about recent events
result = judge(statement="Donald Trump won the 2024 U.S. presidential election.")

print(f"Verdict: {result.overall_verdict}")
print(f"Confidence: {result.confidence}")
print(f"Web Search Used: {result.web_search_performed}")
print(f"Reasoning: {result.reasoning}")
```

### Without Web Search (Original Behavior)

```python
# Create judge with web search disabled
judge = JudgeModule(enable_web_search=False)

# Behaves like original implementation
result = judge(statement="Some statement")
```

## Testing

Run the test script to see the enhancement in action:

```bash
python test_judge_with_search.py
```

The test script evaluates 5 statements:
1. Historical fact (should NOT trigger search)
2. Recent OpenAI event (should trigger search)
3. 2024 election (should trigger search)
4. Clear false statement (should NOT trigger search)
5. Uncertain recent event (should trigger search)

## Expected Impact

### Performance Improvements

1. **Accuracy on 2025 Events**: Instead of returning `CONTAINS_UNSUPPORTED_CLAIMS` (0.5 score), the module can now verify recent events and return definitive verdicts (1.0 score for correct predictions).

2. **Reduced Partial Credit**: Fewer 0.5 scores from the `gepa_metric` function, leading to higher overall evaluation scores.

3. **Broader Coverage**: Can handle statements about:
   - Recent events (2024-2025)
   - Current statistics and facts
   - Breaking news
   - Recent product releases
   - Election results
   - Recent appointments/leadership changes

### Efficiency Considerations

- Web search is **only triggered when needed** (knowledge limitations detected)
- Historical facts and clear true/false statements are handled without search
- Search is limited to top 5 results for quick evaluation
- Can be disabled entirely with `enable_web_search=False` for pure LLM judgment

## Backward Compatibility

The enhancement is fully backward compatible:
- Default behavior includes web search (improved accuracy)
- Original behavior available via `enable_web_search=False`
- Return value includes new `web_search_performed` field (additional info, doesn't break existing code)
- All existing method signatures remain unchanged

## Future Enhancements

Possible improvements for future iterations:

1. **Smart Query Extraction**: Use LLM to extract key entities/terms from statement for more targeted searches
2. **News Search**: Use `SerperService.search_news()` for time-sensitive claims
3. **Multi-Query Search**: Perform multiple searches for complex statements
4. **Confidence Calibration**: Adjust confidence scores based on search result quality
5. **Caching**: Cache search results for repeated statements
6. **Result Ranking**: Score search results by relevance before formatting
7. **Source Quality**: Consider domain authority and publication date in evaluation

## Related Files

- **Metric Function**: `src/optimizer/gepa_optimize.py` (line 46-62, `gepa_metric`)
- **Serper Service**: `src/services/serper_service.py`
- **Base Judge Signature**: `src/factchecker/simple/signatures/judge.py`
- **Evaluation System**: `src/evaluation/evaluate.py`
- **Full Pipeline**: `src/factchecker/modules/fact_checker_pipeline.py` (more comprehensive alternative)

## Architecture Comparison

### Before Enhancement
```
Statement → Judge (LLM) → Verdict
                          (0.5 score for recent events)
```

### After Enhancement
```
Statement → Judge (LLM) → Knowledge Limitations Detected?
                          ↓                    ↓
                          NO                   YES
                          ↓                    ↓
                          Verdict              Web Search → JudgeWithContext → Verdict
                          (0.0 or 1.0)                                         (0.0 or 1.0)
```

## Summary

The enhanced `JudgeModule` bridges the gap between pure LLM knowledge and real-world verifiable facts by selectively incorporating web search when the LLM indicates uncertainty. This addresses the systematic 0.5 scoring issue for recent events while maintaining efficiency for statements that don't require external verification.
