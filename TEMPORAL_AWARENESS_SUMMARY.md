# Temporal Awareness Enhancement for JudgeModule

## Summary

Added temporal awareness to the `JudgeModule` class in `src/factchecker/simple/modules/judge_module.py` to automatically detect time-sensitive claims and trigger web search for verification, preventing false SUPPORTED verdicts based on outdated training data.

## Changes Made

### 1. Added Dependencies
- Imported `datetime` from the standard library
- Imported `relativedelta` from `dateutil` (already in requirements.txt)

### 2. Modified `_detect_knowledge_limitations` Method
- Added a new `statement` parameter (default: empty string)
- Integrated temporal reference detection by calling `_extract_temporal_references(statement)`
- Now triggers web search if temporal references requiring verification are detected
- Updated the method call in `forward()` to pass the statement

### 3. Added `_extract_temporal_references` Helper Method

This new method detects temporal references using multiple patterns:

#### Date Patterns Detected:
1. **YYYY-MM-DD format**: e.g., "2024-06-15"
2. **Month YYYY format**: e.g., "January 2024", "Jan 2024"
3. **"in 20XX" format**: e.g., "in 2024"
4. **Standalone years**: e.g., "2024" (uses Dec 31 for inclusive matching)

#### Temporal Keywords Detected:
- recent, recently
- latest
- current, currently
- this year, last year
- last month, this month
- today, now
- present
- up-to-date, up to date
- modern
- ongoing
- as of

#### Logic:
- Calculates a 24-month cutoff date from today
- Extracts all dates and temporal keywords from the statement
- Triggers web search if:
  - ANY detected date is within the last 24 months, OR
  - ANY temporal keyword is present

#### Returns:
```python
{
    'dates': [datetime objects],
    'temporal_keywords': [list of keywords],
    'needs_verification': bool
}
```

## Test Results

All test cases pass successfully:

### Temporal Extraction Tests (16 cases):
✓ Recent dates (2024, 2025) trigger web search
✓ Temporal keywords ("recent", "latest", "current", etc.) trigger web search
✓ Old dates (1945, 1980s, 1600) do NOT trigger
✓ Non-temporal statements do NOT trigger

### Knowledge Limitation Detection Tests (4 cases):
✓ Statement with temporal reference + SUPPORTED verdict → triggers
✓ Non-temporal fact → does not trigger
✓ Temporal keyword + UNSUPPORTED verdict → triggers
✓ Historical fact → does not trigger

## Benefits

1. **Prevents False Positives**: Statements about recent events won't be marked as SUPPORTED based solely on outdated training data
2. **Automatic Detection**: No manual intervention needed - the system automatically identifies time-sensitive claims
3. **Configurable Threshold**: 24-month cutoff can be easily adjusted by modifying the `relativedelta(months=24)` parameter
4. **Comprehensive Coverage**: Detects both explicit dates and implicit temporal indicators
5. **Backward Compatible**: Existing functionality preserved; only adds additional trigger conditions

## Example Usage

```python
from src.factchecker.simple.modules.judge_module import JudgeModule

judge = JudgeModule(enable_web_search=True)

# This will trigger web search due to temporal reference
result = judge.forward("The recent elections in 2025 showed high turnout.")
print(result.web_search_performed)  # True

# This will NOT trigger web search (no temporal reference)
result = judge.forward("Water boils at 100 degrees Celsius.")
print(result.web_search_performed)  # False
```

## Files Modified

- `src/factchecker/simple/modules/judge_module.py` - Main implementation

## Files Created

- `test_temporal_awareness.py` - Comprehensive test suite
- `TEMPORAL_AWARENESS_SUMMARY.md` - This documentation

## Future Enhancements

Potential improvements for future versions:

1. **Relative Date Parsing**: Handle phrases like "two weeks ago", "next quarter"
2. **Configurable Cutoff**: Allow users to set custom time windows
3. **Domain-Specific Patterns**: Different cutoffs for different claim types (e.g., sports scores vs. scientific facts)
4. **Fuzzy Date Matching**: Handle informal date expressions like "early 2024", "mid-2023"
5. **Multi-language Support**: Extend patterns to support non-English temporal references
