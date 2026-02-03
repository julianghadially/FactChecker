# JudgeModule Enhancement Summary

## Overview
Enhanced the `JudgeModule` in `src/factchecker/simple/modules/judge_module.py` to add lightweight web search capability when the LLM detects knowledge cutoff limitations.

## Changes Made

### 1. **Enhanced JudgeModule Architecture**

#### Two-Stage Evaluation Process
1. **Stage 1**: Initial judgment using parametric knowledge via `ChainOfThought(Judge)`
2. **Stage 2**: If reasoning indicates uncertainty due to temporal/knowledge limitations:
   - Detect knowledge limitations using keyword matching
   - Trigger focused web search using `SerperService`
   - Scrape 1-2 top results using `FirecrawlService`
   - Re-evaluate with evidence appended to statement context

#### New Features
- **`use_web_search` parameter**: Added to `__init__()` (default: `True`) to allow disabling web search
- **Lazy initialization**: Web services are only initialized when needed
- **Knowledge limitation detection**: Monitors reasoning for keywords indicating uncertainty
- **Graceful fallback**: If web search fails, returns original judgment

### 2. **Knowledge Limitation Detection**

The module detects uncertainty through keyword matching in the LLM's reasoning:

```python
UNCERTAINTY_KEYWORDS = [
    "knowledge cutoff",
    "cannot verify",
    "after my training",
    "do not have",
    "don't have",
    "unable to verify",
    "no information",
    "lack information",
    "beyond my knowledge",
    "outside my knowledge",
    "recent event",
    "recent information",
    "as of my",
    "training data",
    "cannot confirm",
    "can't confirm",
]
```

### 3. **Web Evidence Gathering**

The `_gather_web_evidence()` method:
- Performs Google search via `SerperService` (default: 2 results)
- Scrapes each result using `FirecrawlService` (max 5000 chars per page)
- Falls back to search snippets if scraping fails
- Formats evidence as structured markdown with source attribution
- Handles errors gracefully

### 4. **Updated Judge Signature**

Modified `src/factchecker/simple/signatures/judge.py` docstring to reflect:
- Ability to use both parametric knowledge and web evidence
- Instruction to clearly indicate knowledge limitations
- Guidance on incorporating web evidence when provided

### 5. **New Output Field**

The `forward()` method now returns:
- `statement`: The input statement
- `overall_verdict`: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
- `confidence`: Float between 0.0 and 1.0
- `reasoning`: Explanation of the verdict
- **`web_evidence_used`**: Boolean indicating if web search was performed (NEW)

## Usage Examples

### Basic Usage (Web Search Enabled)
```python
from src.factchecker.simple.modules.judge_module import JudgeModule

# Initialize with web search enabled (default)
judge = JudgeModule(use_web_search=True)

# Evaluate a recent event
result = judge(statement="SpaceX launched Starship Flight 6 in November 2024.")

print(f"Verdict: {result.overall_verdict}")
print(f"Web Evidence Used: {result.web_evidence_used}")
print(f"Reasoning: {result.reasoning}")
```

### Disable Web Search
```python
# Initialize with web search disabled
judge = JudgeModule(use_web_search=False)

# Will only use parametric knowledge
result = judge(statement="The 2024 US Presidential election was held in November.")
```

### Historical Facts (No Web Search Needed)
```python
judge = JudgeModule()

# This won't trigger web search as it's within LLM knowledge
result = judge(statement="The Earth orbits around the Sun.")
print(f"Web Evidence Used: {result.web_evidence_used}")  # False
```

## Key Design Decisions

### 1. **Maintains Simplicity**
- Minimal additional complexity to the existing architecture
- Web services initialized lazily only when needed
- Can be completely disabled via parameter

### 2. **Smart Triggering**
- Only performs web search when LLM explicitly indicates uncertainty
- Avoids unnecessary API calls for well-known facts
- Keyword-based detection is simple but effective

### 3. **Cost-Conscious**
- Limits to 2 search results by default (configurable)
- Truncates scraped content to 5000 chars per page
- Lazy initialization of services

### 4. **Robust Error Handling**
- Graceful degradation if web search fails
- Falls back to search snippets if scraping fails
- Returns original judgment if web services unavailable

### 5. **Transparency**
- `web_evidence_used` flag clearly indicates when web search occurred
- Evidence is clearly marked in the statement context
- Maintains original reasoning in output

## Files Modified

1. **`src/factchecker/simple/modules/judge_module.py`**
   - Added web search capability
   - Implemented two-stage architecture
   - Added knowledge limitation detection
   - Integrated SerperService and FirecrawlService

2. **`src/factchecker/simple/signatures/judge.py`**
   - Updated docstring to reflect web evidence capability
   - Added guidance for indicating knowledge limitations

## Dependencies

The enhancement relies on existing services:
- `src.services.serper_service.SerperService` - Google search via Serper API
- `src.services.firecrawl_service.FirecrawlService` - Web page scraping

Requires environment variables:
- `SERPER_API_KEY` - Serper API key
- `FIRECRAWL_API_KEY` - Firecrawl API key

## Testing Recommendations

1. **Test with historical facts**: Verify web search is NOT triggered
2. **Test with recent events**: Verify web search IS triggered
3. **Test with `use_web_search=False`**: Verify web search never occurs
4. **Test error handling**: Simulate API failures
5. **Test edge cases**: Very long statements, ambiguous statements

## Future Enhancements

Potential improvements for consideration:
1. **LLM-based query generation**: Use LLM to derive better search queries
2. **Configurable max_results**: Allow user to specify number of results
3. **News vs. regular search**: Automatically choose news search for temporal queries
4. **Evidence caching**: Cache scraped content to reduce API calls
5. **Confidence threshold**: Trigger web search based on confidence score
6. **Structured evidence parsing**: Extract specific claims from evidence
7. **Multi-turn refinement**: Iteratively search if first attempt insufficient

## Migration Notes

### Backward Compatibility
- **Fully backward compatible**: Existing code works without changes
- Default behavior includes web search, but can be disabled
- Output includes new `web_evidence_used` field, but existing fields unchanged

### API Signature Changes
```python
# Old initialization (still works)
judge = JudgeModule()

# New initialization with web search disabled
judge = JudgeModule(use_web_search=False)

# New output field
result = judge(statement="...")
web_used = result.web_evidence_used  # New field
```

## Performance Considerations

### Latency
- **No web search**: Same as before (~1-2 seconds for LLM call)
- **With web search**: Additional 5-10 seconds for search + scraping
  - Serper search: ~1-2 seconds
  - Firecrawl scraping (2 pages): ~4-8 seconds total
  - Second LLM judgment: ~1-2 seconds

### Cost
- **No web search**: Same as before (1 LLM call)
- **With web search**: Higher cost
  - 2 LLM calls (initial + re-evaluation)
  - 1 Serper search (~$0.001 per search)
  - 2 Firecrawl scrapes (~$0.002-0.004 per scrape)

### Rate Limits
- Respects existing service rate limits
- Consider implementing caching for frequently checked statements

## Conclusion

This enhancement successfully adds lightweight web search capability to the `JudgeModule` while maintaining its simplicity and ease of use. The two-stage architecture ensures that web search is only triggered when necessary, minimizing cost and latency while enabling verification of recent events and information beyond the LLM's knowledge cutoff.
