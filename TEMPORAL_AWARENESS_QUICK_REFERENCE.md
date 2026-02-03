# Temporal Awareness - Quick Reference

## What Is It?
Automatically triggers web search for time-sensitive claims to prevent false verdicts based on outdated LLM training data.

## How It Works

### Triggers Web Search When Statement Contains:

✓ **Recent Dates** (within 24 months)
- `2024-06-15`
- `January 2025`
- `in 2024`
- `2025`

✓ **Temporal Keywords**
- recent, recently, latest
- current, currently
- this year, last year, last month
- today, now, present
- up-to-date, ongoing

✗ **Does NOT Trigger For:**
- Historical dates (> 24 months old)
- Timeless facts
- No temporal references

## Quick Start

```python
from src.factchecker.simple.modules.judge_module import JudgeModule

# Enable temporal awareness (default behavior)
judge = JudgeModule(enable_web_search=True)

# Automatic temporal detection
result = judge.forward("Recent studies from 2024 show results.")
print(result.web_search_performed)  # True

result = judge.forward("Water boils at 100°C.")
print(result.web_search_performed)  # False
```

## Key Functions

### `forward(statement)`
Main method - automatically detects temporal references and triggers web search if needed.

### `_extract_temporal_references(statement)`
Returns:
```python
{
    'dates': [datetime, ...],
    'temporal_keywords': ['recent', ...],
    'needs_verification': True/False
}
```

### `_detect_knowledge_limitations(reasoning, verdict, statement="")`
Enhanced to check temporal references in statement.

## Examples

| Statement | Triggers Web Search? | Why? |
|-----------|---------------------|------|
| "The 2025 elections showed record turnout" | ✓ Yes | Contains 2025 (within 24 months) |
| "Recent research confirms the theory" | ✓ Yes | Contains "recent" keyword |
| "The current president announced..." | ✓ Yes | Contains "current" keyword |
| "World War II ended in 1945" | ✗ No | Historical date (> 24 months) |
| "Paris is the capital of France" | ✗ No | No temporal reference |

## Configuration

### Change Time Window
```python
# In _extract_temporal_references() method:
cutoff_date = today - relativedelta(months=24)  # Default: 24 months
cutoff_date = today - relativedelta(months=12)  # Change to 12 months
```

### Add Custom Keywords
```python
# In _extract_temporal_references() method:
temporal_keyword_patterns = [
    r'\brecent\b',
    r'\byour_keyword\b',  # Add here
]
```

## Testing

```bash
# Unit tests
python test_temporal_awareness.py

# Real-world examples
python test_real_world_example.py

# Integration test (requires API keys)
python test_judge_with_search.py
```

## Common Issues

**Web search not triggering?**
1. Check: `enable_web_search=True`
2. Check: Date within 24 months?
3. Check: Pattern recognized?

**False positives?**
- Refine regex patterns
- Add word boundaries: `\bkeyword\b`

## Files Modified
- `src/factchecker/simple/modules/judge_module.py`

## Files Created
- `test_temporal_awareness.py` - Test suite
- `test_real_world_example.py` - Examples
- `docs/TEMPORAL_AWARENESS_FEATURE.md` - Full docs
- `TEMPORAL_AWARENESS_SUMMARY.md` - Implementation summary

## Key Benefits
1. ✓ Prevents false SUPPORTED verdicts on recent claims
2. ✓ Automatic detection - no manual intervention
3. ✓ Backward compatible - no breaking changes
4. ✓ Configurable 24-month window
5. ✓ Comprehensive pattern matching

## Pattern Matching Details

### Date Formats Detected
- `YYYY-MM-DD`: 2024-06-15
- `Month YYYY`: January 2024, Jan 2024
- `in YYYY`: in 2024, in 2025
- `Year only`: 2024, 2025

### Temporal Keywords (Full List)
```
recent, recently, latest, current, currently,
this year, last year, last month, this month,
today, now, present, up-to-date, up to date,
modern, ongoing, as of
```

## Dependencies
- ✓ `datetime` (standard library)
- ✓ `python-dateutil` (in requirements.txt)
- ✓ `re` (standard library)

## No Migration Needed!
Existing code works without changes. Temporal awareness is automatic when `enable_web_search=True`.
