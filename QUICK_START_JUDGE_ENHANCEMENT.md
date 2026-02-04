# Quick Start: Enhanced JudgeModule

## TL;DR

The `JudgeModule` now automatically searches the web when it's uncertain about a statement, solving the "knowledge cutoff" problem.

## Basic Usage

```python
from src.factchecker.simple.modules.judge_module import JudgeModule

judge = JudgeModule()

# Will automatically use web search if needed
result = judge.forward("Donald Trump won the 2024 U.S. presidential election")

print(f"Verdict: {result.overall_verdict}")
print(f"Confidence: {result.confidence}")
print(f"Used web search: {result.used_web_search}")
```

## When Does It Use Web Search?

Web search triggers when **EITHER** condition is true:

1. **Low confidence**: `confidence < 0.7`
2. **Uncertainty keywords** in reasoning:
   - "knowledge cutoff"
   - "lacking information"
   - "unable to verify"
   - "cannot confirm"
   - "no current information"
   - etc.

## Output Fields

```python
result.statement          # Input statement
result.overall_verdict    # SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
result.confidence         # Float 0.0-1.0
result.reasoning          # Explanation
result.used_web_search    # True if web search was triggered
result.evidence           # Retrieved evidence (only if web search used)
```

## Disable Web Search (Original Behavior)

```python
result = judge.forward(statement, web_search_enabled=False)
```

## Performance

| Scenario | Speed | Cost |
|----------|-------|------|
| Historical fact (no web search) | ~2 sec | 1 LLM call |
| Recent event (web search) | ~15 sec | 2 LLM calls + 3 scrapes |

## Test It

```bash
python test_judge_enhancement.py
```

## Architecture

```
Statement → LLM Judge → Uncertain? → Yes → Web Search → Re-judge → Result
                     ↓ No
                     └────────────────────────────────────→ Result
```

## Key Files

- **Module**: `src/factchecker/simple/modules/judge_module.py`
- **Signatures**:
  - `src/factchecker/simple/signatures/judge.py` (LLM-only)
  - `src/factchecker/simple/signatures/web_augmented_judge.py` (with evidence)
- **Test**: `test_judge_enhancement.py`
- **Docs**: `JUDGE_MODULE_ENHANCEMENT.md`

## Examples

### Example 1: Historical Fact (No Web Search)
```python
statement = "The United States declared independence in 1776"
result = judge.forward(statement)
# result.used_web_search = False (confident, no search needed)
```

### Example 2: Recent Event (Web Search Triggered)
```python
statement = "Donald Trump won the 2024 U.S. presidential election"
result = judge.forward(statement)
# result.used_web_search = True (mentions "knowledge cutoff")
```

### Example 3: False Recent Claim (Web Search Triggered)
```python
statement = "OpenAI released GPT-5 in 2025"
result = judge.forward(statement)
# result.used_web_search = True (low confidence, searches and refutes)
```

## What Changed?

### New Capabilities
✅ Can verify recent events (2024-2025)
✅ Automatically searches web when uncertain
✅ Scrapes 2-3 sources for evidence
✅ Transparent decision making (shows if web search was used)

### Backward Compatible
✅ Existing code works without changes
✅ Can disable with `web_search_enabled=False`
✅ Same output schema (with extra fields)

## Dependencies

- **SerperService**: Google search
- **FirecrawlService**: Web scraping
- **DSPy**: LLM framework
- **re**: Pattern matching

## Questions?

See full documentation: `JUDGE_MODULE_ENHANCEMENT.md`
