# JudgeModule Enhancement: URL Evidence Support

## Overview

The simple `JudgeModule` has been enhanced to accept and utilize evidence from provided URLs. This enables evidence-based verification against actual source material, while maintaining backward compatibility for cases without URLs.

## Motivation

The enhancement directly addresses the limitation where the JudgeModule would score low (e.g., 0.5) on recent 2025 events due to relying solely on the LLM's outdated knowledge cutoff. With URL support, the module can now verify claims against actual web sources, significantly improving accuracy on current events.

## Changes Made

### 1. Judge Signature Update (`src/factchecker/simple/signatures/judge.py`)

Added an optional `evidence` input field:

```python
evidence: str = InputField(
    desc="External evidence from web sources (leave empty if unavailable)",
    default=""
)
```

This field allows the LLM to consider external evidence when making its verdict, but defaults to empty string for backward compatibility.

### 2. JudgeModule Enhancement (`src/factchecker/simple/modules/judge_module.py`)

#### Added Dependencies
- `Optional` from typing for optional parameters
- `FirecrawlService` for web scraping

#### Updated `__init__` Method
- Initializes `FirecrawlService` instance

#### Enhanced `forward()` Method

**New Parameter:**
- `url: Optional[str] = None` - Can accept:
  - `None` - No evidence scraping (backward compatible)
  - Single URL string - Scrapes one source
  - Comma-separated URLs - Scrapes multiple sources

**Scraping Logic:**
1. If URL(s) provided, splits by comma and processes each
2. Uses `FirecrawlService.scrape()` to fetch markdown content
3. Aggregates successful scrapes with source labels
4. Handles failures gracefully (logs warning, continues)
5. Passes aggregated evidence to the judge signature

**Error Handling:**
- Catches scraping failures and logs warnings
- Falls back to knowledge-only judgment if all scrapes fail
- Maintains functionality even with invalid/unreachable URLs

## Usage Examples

### Backward Compatible (No URL)
```python
from src.factchecker.simple import JudgeModule

judge = JudgeModule()
result = judge.forward(
    statement="The United States has the highest number of nuclear power plants"
)
# Uses LLM knowledge only
```

### Single URL Evidence
```python
judge = JudgeModule()
result = judge.forward(
    statement="Alaska Airlines is launching London flights in May 2026",
    url="https://thepointsguy.com/news/alaska-airlines-london-heathrow-seattle-nonstop-flights/"
)
# Scrapes URL and uses content as evidence
```

### Multiple URLs
```python
judge = JudgeModule()
result = judge.forward(
    statement="Alaska Airlines is launching London flights",
    url="https://url1.com,https://url2.com"
)
# Scrapes both URLs and aggregates evidence
```

### Graceful Failure
```python
judge = JudgeModule()
result = judge.forward(
    statement="The sky is blue",
    url="https://invalid-url-that-does-not-exist.com"
)
# Logs warning, falls back to knowledge-only judgment
```

## Benefits

1. **Evidence-Based Verification**: Can verify claims against actual source material
2. **Improved Accuracy on Recent Events**: No longer limited by LLM knowledge cutoff
3. **Backward Compatible**: Existing code continues to work without changes
4. **Flexible**: Supports single or multiple evidence sources
5. **Robust**: Gracefully handles scraping failures
6. **Cost-Effective**: Only scrapes when URLs are provided

## Integration with Evaluation Data

The enhancement is designed to work seamlessly with evaluation datasets that include URL fields (like the fortune500 news articles data). The evaluation pipeline can now pass URLs to leverage actual source evidence:

```python
# Example evaluation with URL
for item in evaluation_data:
    result = judge.forward(
        statement=item['claim'],
        url=item.get('url', None)  # Use URL if available
    )
```

## Testing

A test script is provided at `/workspace/test_judge_module_enhancement.py` with test cases for:
1. Without URL (backward compatibility)
2. With single URL
3. With multiple URLs
4. With invalid URL (graceful failure)

To run tests:
```bash
python test_judge_module_enhancement.py
```

## Technical Notes

### FirecrawlService Integration
- Uses existing `FirecrawlService` from `src/services/firecrawl_service.py`
- Respects max_length parameter (default 10000 chars) to manage token costs
- Handles PDF documents and content truncation
- Returns markdown-formatted content for optimal LLM processing

### Evidence Formatting
Evidence is formatted with clear source labels:
```
--- Evidence from https://example.com ---
[scraped markdown content]

--- Evidence from https://example2.com ---
[scraped markdown content]
```

This helps the LLM understand the source of each piece of evidence and maintain context.

### Performance Considerations
- Scraping adds latency (typically 1-3 seconds per URL)
- Consider batching evaluation jobs or using async patterns for large-scale evaluation
- Failed scrapes log warnings but don't block execution
- FirecrawlService has built-in rate limiting and error handling

## Future Enhancements

Potential improvements for future iterations:
1. Async scraping for better performance with multiple URLs
2. Caching mechanism to avoid re-scraping the same URLs
3. Configurable max_length per scrape
4. Support for passing pre-scraped content directly
5. Metrics tracking for scrape success rates
6. Retry logic for transient failures

## Conclusion

This enhancement transforms the simple JudgeModule from a knowledge-only fact checker into a hybrid system that can leverage external evidence when available, while maintaining its simplicity and backward compatibility. It bridges the gap between the lightweight simple judge and the full research-capable FactCheckerPipeline.
