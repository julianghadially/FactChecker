# Temporal Awareness Feature Documentation

## Overview

The temporal awareness feature enhances the `JudgeModule` to automatically detect time-sensitive claims and trigger web search for verification. This prevents false SUPPORTED verdicts based on outdated training data when evaluating recent or current events.

## Problem Statement

Large Language Models (LLMs) have a knowledge cutoff date and cannot reliably verify claims about recent events, current data, or time-sensitive information. Without temporal awareness, the fact-checker might incorrectly mark recent claims as SUPPORTED based on:
- Outdated training data
- Pattern matching from similar historical events
- Speculation or hallucination about recent events

## Solution

The enhanced `JudgeModule` now automatically detects temporal references in statements and triggers web search when:
1. The statement contains dates within the last 24 months
2. The statement contains temporal keywords (e.g., "recent", "latest", "current")

## Architecture

### Modified Components

#### 1. `_detect_knowledge_limitations(reasoning, verdict, statement="")`

**Enhanced Signature:**
```python
def _detect_knowledge_limitations(self, reasoning: str, verdict: str, statement: str = "") -> bool
```

**New Behavior:**
- Added optional `statement` parameter (maintains backward compatibility)
- Calls `_extract_temporal_references(statement)` to analyze temporal content
- Returns `True` if temporal verification is needed

**Backward Compatibility:**
- The `statement` parameter defaults to empty string
- Old code calling without `statement` continues to work
- Only the `forward()` method's internal call includes the statement

#### 2. `_extract_temporal_references(statement)` *(NEW)*

**Purpose:** Comprehensive temporal reference detection

**Date Pattern Detection:**
1. **YYYY-MM-DD format**: `2024-06-15`
2. **Month YYYY format**: `January 2024`, `Jan 2024`
3. **"in YYYY" format**: `in 2024`
4. **Standalone years**: `2024`, `2025`

**Temporal Keyword Detection:**
- `recent`, `recently`
- `latest`
- `current`, `currently`
- `this year`, `last year`
- `last month`, `this month`
- `today`, `now`
- `present`
- `up-to-date`, `up to date`
- `modern`
- `ongoing`
- `as of`

**Return Value:**
```python
{
    'dates': List[datetime],           # Detected date objects
    'temporal_keywords': List[str],    # Detected keywords
    'needs_verification': bool         # Whether web search is needed
}
```

**Logic:**
- Calculates 24-month cutoff date from current date
- For year-only patterns, uses December 31st (inclusive matching)
- Triggers verification if ANY date is within 24 months OR ANY temporal keyword is present

## Usage Examples

### Basic Usage

```python
from src.factchecker.simple.modules.judge_module import JudgeModule

# Create judge with web search enabled
judge = JudgeModule(enable_web_search=True)

# Recent claim - will trigger web search
result = judge.forward("The 2025 elections showed record turnout.")
print(f"Web search performed: {result.web_search_performed}")  # True

# Historical fact - no web search needed
result = judge.forward("World War II ended in 1945.")
print(f"Web search performed: {result.web_search_performed}")  # False
```

### Direct Temporal Analysis

```python
from src.factchecker.simple.modules.judge_module import JudgeModule

judge = JudgeModule(enable_web_search=False)

# Analyze temporal content
statement = "Recent studies from January 2024 show promising results."
temporal_info = judge._extract_temporal_references(statement)

print(f"Dates found: {temporal_info['dates']}")
print(f"Keywords found: {temporal_info['temporal_keywords']}")
print(f"Needs verification: {temporal_info['needs_verification']}")
```

### Custom Web Search Handling

```python
from src.factchecker.simple.modules.judge_module import JudgeModule

# Disable automatic web search
judge = JudgeModule(enable_web_search=False)

statement = "The current market conditions favor investors."

# Check if web search would be needed
if judge._detect_knowledge_limitations("", "SUPPORTED", statement):
    print("This claim requires web verification")
    # Implement custom search logic here
```

## Configuration

### Adjusting the Time Window

The 24-month cutoff is configured in `_extract_temporal_references()`:

```python
cutoff_date = today - relativedelta(months=24)  # Current: 24 months
```

To modify:
```python
cutoff_date = today - relativedelta(months=12)  # 12 months
cutoff_date = today - relativedelta(months=36)  # 36 months
```

### Adding Custom Temporal Keywords

Add patterns to the `temporal_keyword_patterns` list:

```python
temporal_keyword_patterns = [
    r'\brecent\b',
    r'\byour_keyword\b',  # Add your pattern here
    # ... existing patterns
]
```

### Adding Custom Date Patterns

Add new pattern detection blocks in `_extract_temporal_references()`:

```python
# Pattern 5: Your custom format
custom_pattern = r'your_regex_pattern'
for match in re.finditer(custom_pattern, statement):
    try:
        date_obj = datetime.strptime(match.group(0), 'your_format')
        dates.append(date_obj)
    except ValueError:
        pass
```

## Test Suite

### Running Tests

```bash
# Basic temporal extraction tests
python test_temporal_awareness.py

# Real-world scenario demonstrations
python test_real_world_example.py

# Integration test with actual web search (requires API keys)
python test_judge_with_search.py
```

### Test Coverage

The test suite validates:
1. ✓ Date pattern detection (YYYY-MM-DD, Month YYYY, etc.)
2. ✓ Temporal keyword detection
3. ✓ 24-month cutoff logic
4. ✓ Integration with knowledge limitation detection
5. ✓ Backward compatibility
6. ✓ Real-world scenarios

## Performance Considerations

### Computational Cost
- **Regex matching**: Minimal overhead (~milliseconds per statement)
- **Date parsing**: Negligible impact for typical statement lengths
- **Web search**: Only triggered when temporal references detected (reduces unnecessary API calls)

### Optimization Tips
1. **Precompile regex patterns** for repeated use (future enhancement)
2. **Cache temporal analysis** for duplicate statements
3. **Adjust time window** to balance accuracy vs. search frequency

## Limitations & Future Enhancements

### Current Limitations
1. Only supports English temporal references
2. Cannot parse relative dates like "two weeks ago"
3. Fixed 24-month window (not configurable per-domain)
4. No handling of fuzzy dates like "early 2024"

### Planned Enhancements
1. **Relative date parsing**: Support "X days/weeks/months ago"
2. **Configurable cutoffs**: Domain-specific time windows
3. **Multi-language support**: Temporal detection for other languages
4. **Fuzzy matching**: Handle "early 2024", "mid-2023", etc.
5. **Performance optimization**: Precompiled regex patterns
6. **Confidence scoring**: Grade urgency of temporal verification

## Dependencies

### Required
- `datetime` (standard library)
- `python-dateutil` (already in requirements.txt)
- `re` (standard library)

### Version Requirements
- Python 3.7+
- python-dateutil 2.8.0+

## Backward Compatibility

### API Changes
- ✓ No breaking changes to public API
- ✓ `_detect_knowledge_limitations()` signature extended with optional parameter
- ✓ Existing code continues to work without modifications
- ✓ New functionality is opt-in (requires passing statement parameter)

### Migration Guide
No migration needed! Existing code works as-is. To leverage temporal awareness:

**Before:**
```python
result = judge.forward(statement)
```

**After:**
```python
result = judge.forward(statement)  # Same call - temporal awareness automatic!
```

## Troubleshooting

### Web Search Not Triggering

**Issue:** Statement contains temporal reference but web search doesn't trigger

**Possible Causes:**
1. `enable_web_search=False` in JudgeModule initialization
2. Date is older than 24-month cutoff
3. Pattern not recognized by regex

**Solution:**
```python
# Check temporal detection
temporal_info = judge._extract_temporal_references(statement)
print(temporal_info)

# Verify web search is enabled
judge = JudgeModule(enable_web_search=True)
```

### False Positives

**Issue:** Non-temporal statements trigger web search

**Possible Causes:**
1. Unintended keyword matches (e.g., "present" as verb vs. time reference)
2. Year numbers in non-temporal context (e.g., "Room 2024")

**Solution:**
- Refine regex patterns to use word boundaries: `r'\bkeyword\b'`
- Add context-aware filtering (future enhancement)

### Performance Issues

**Issue:** Temporal analysis taking too long

**Possible Causes:**
1. Very long statements
2. Complex regex patterns on large text

**Solution:**
- Truncate statements to reasonable length before analysis
- Consider caching results for duplicate statements

## Support & Contributing

### Reporting Issues
- Check existing test cases to verify expected behavior
- Include statement text and temporal analysis output
- Note Python version and dependency versions

### Contributing Enhancements
1. Add test cases for new patterns
2. Document changes in this file
3. Ensure backward compatibility
4. Run full test suite before submitting

## References

### Related Files
- `src/factchecker/simple/modules/judge_module.py` - Main implementation
- `test_temporal_awareness.py` - Test suite
- `test_real_world_example.py` - Usage examples
- `TEMPORAL_AWARENESS_SUMMARY.md` - Quick reference

### External Documentation
- [python-dateutil](https://dateutil.readthedocs.io/) - Date handling library
- [Python regex](https://docs.python.org/3/library/re.html) - Regular expressions
- [DSPy Framework](https://dspy-docs.vercel.app/) - LLM pipeline framework
