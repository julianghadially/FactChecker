# JudgeModule Enhancement - Quick Reference

## What Changed?

The `JudgeModule` now automatically performs web search when it detects knowledge cutoff limitations in its reasoning.

## Quick Start

### Basic Usage (Web Search Enabled)
```python
from src.factchecker.simple.modules.judge_module import JudgeModule

judge = JudgeModule()  # Web search enabled by default
result = judge(statement="Your statement here")

print(f"Verdict: {result.overall_verdict}")
print(f"Web Search Used: {result.web_evidence_used}")
```

### Disable Web Search
```python
judge = JudgeModule(use_web_search=False)  # Old behavior
result = judge(statement="Your statement here")
```

## Output Fields

```python
result = judge(statement="...")

# Available fields:
result.statement          # Input statement
result.overall_verdict    # SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
result.confidence         # Float 0.0-1.0
result.reasoning          # Explanation
result.web_evidence_used  # Boolean (NEW)
```

## When Does Web Search Trigger?

Web search triggers when the LLM's reasoning contains phrases like:
- "knowledge cutoff"
- "cannot verify"
- "after my training"
- "do not have"
- "recent event"
- "cannot confirm"
- ... and 10 more keywords

## Configuration

```python
# Default: Web search enabled
judge = JudgeModule()

# Disable web search
judge = JudgeModule(use_web_search=False)
```

## Environment Requirements

Ensure these environment variables are set:
```bash
export SERPER_API_KEY="your_serper_key"
export FIRECRAWL_API_KEY="your_firecrawl_key"
```

## Performance

- **Without web search**: ~1-2 seconds (1 LLM call)
- **With web search**: ~6-12 seconds (2 LLM calls + search + scraping)

## Cost

- **Without web search**: Standard LLM cost
- **With web search**: +~$0.01-0.02 per statement
  - Extra LLM call
  - 1 Serper search
  - 2 Firecrawl scrapes

## Examples

### Historical Fact (No Web Search)
```python
judge = JudgeModule()
result = judge(statement="The Earth orbits around the Sun.")

# Result:
# verdict: SUPPORTED
# web_evidence_used: False
# (No web search needed)
```

### Recent Event (Web Search Triggered)
```python
judge = JudgeModule()
result = judge(statement="SpaceX launched Starship in 2024.")

# Result:
# verdict: SUPPORTED (based on web evidence)
# web_evidence_used: True
# (Web search performed automatically)
```

## Troubleshooting

### Web search not triggering when expected?
- Check that `use_web_search=True` (default)
- Verify LLM is expressing uncertainty in reasoning
- Check API keys are set correctly

### Web search triggering too often?
- Set `use_web_search=False` to disable
- Or modify `UNCERTAINTY_KEYWORDS` list in the module

### API errors?
- Verify Serper API key is valid
- Verify Firecrawl API key is valid
- Check API rate limits

## Files Modified

- `src/factchecker/simple/modules/judge_module.py` - Main module
- `src/factchecker/simple/signatures/judge.py` - Signature docstring

## Backward Compatibility

✅ **Fully backward compatible** - existing code works without changes!

```python
# Old code still works:
judge = JudgeModule()
result = judge(statement="...")
# All existing fields available + new web_evidence_used field
```

## Testing

Run the test script:
```bash
python test_judge_enhancement.py
```

## Further Reading

- `JUDGE_MODULE_ENHANCEMENT_SUMMARY.md` - Detailed technical summary
- `BEFORE_AFTER_COMPARISON.md` - Comprehensive before/after comparison
- `example_judge_with_web_search.py` - Usage examples

## Support

For issues or questions:
1. Check environment variables are set
2. Verify API keys are valid
3. Review error messages in logs
4. Check `web_evidence_used` flag to see if search was attempted
