# AdaptiveJudgeModule - Quick Start Guide

## 30-Second Overview

`AdaptiveJudgeModule` intelligently routes fact-checking between fast LLM-only verification and thorough web research based on confidence levels.

```python
from src.factchecker.modules import AdaptiveJudgeModule

# Initialize
adaptive_judge = AdaptiveJudgeModule()

# Use
result = adaptive_judge(statement="Your statement here")

# Check
print(f"Verdict: {result.overall_verdict}")
print(f"Confidence: {result.confidence}")
print(f"Research performed: {result.fallback_triggered}")
```

## When Fallback Triggers

Automatic web research happens when **ALL** conditions are met:

1. ✅ Verdict is `"CONTAINS_UNSUPPORTED_CLAIMS"`
2. ✅ Confidence < threshold (default: 0.7)
3. ✅ Fallback enabled (default: True)

## Setup

### 1. Install Dependencies (if not already)
```bash
pip install dspy-ai openai  # Or other LLM provider
pip install firecrawl-py    # For web scraping (optional, only for fallback)
# Serper API for search (optional, only for fallback)
```

### 2. Set API Keys
```bash
# Required always
export OPENAI_API_KEY="your-openai-key"

# Required only when fallback triggers
export SERPER_API_KEY="your-serper-key"
export FIRECRAWL_API_KEY="your-firecrawl-key"
```

### 3. Configure DSPy
```python
import dspy
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)
```

## Basic Usage

```python
from src.factchecker.modules import AdaptiveJudgeModule

# Default settings
adaptive_judge = AdaptiveJudgeModule()

# Fact-check a statement
result = adaptive_judge(statement="The Earth orbits the Sun.")

# Access results
print(f"Verdict: {result.overall_verdict}")
# → "SUPPORTED"

print(f"Confidence: {result.confidence:.2f}")
# → 0.95

print(f"Reasoning: {result.reasoning}")
# → "This is a well-established scientific fact..."

print(f"Fallback triggered: {result.fallback_triggered}")
# → False (high confidence, no research needed)
```

## Configuration Examples

### Conservative (More Research)
```python
adaptive_judge = AdaptiveJudgeModule(
    confidence_threshold=0.9,  # Trigger fallback more often
    max_judge_iterations=5,    # More thorough research
    max_page_visits=5
)
```
**Use for**: Medical, legal, financial fact-checking

### Aggressive (Less Research)
```python
adaptive_judge = AdaptiveJudgeModule(
    confidence_threshold=0.5,  # Trigger fallback less often
    max_judge_iterations=2,    # Faster research
    max_page_visits=2
)
```
**Use for**: High-volume processing, quick checks

### Judge-Only (No Research)
```python
adaptive_judge = AdaptiveJudgeModule(
    enable_fallback=False  # Never do web research
)
```
**Use for**: Initial screening, cost-sensitive applications

## Understanding Results

### When No Fallback Occurs (Fast Path)
```python
result = adaptive_judge(statement="Water boils at 100°C")

# Available fields:
result.statement            # "Water boils at 100°C"
result.overall_verdict      # "SUPPORTED"
result.confidence           # 0.95
result.reasoning            # "This is a well-known..."
result.fallback_triggered   # False
```

### When Fallback Occurs (Slow Path)
```python
result = adaptive_judge(statement="Company X Q4 2024 revenue was $523M")

# All previous fields plus:
result.claims               # ["Company X Q4 2024 revenue was $523M"]
result.claim_results        # [ClaimResult objects with evidence]

# Access detailed evidence:
for claim_result in result.claim_results:
    print(f"Claim: {claim_result.claim}")
    print(f"Verdict: {claim_result.verdict}")
    print(f"Evidence: {claim_result.evidence_summary}")
```

## Output Format

### Verdict Types
| Verdict | Meaning |
|---------|---------|
| `SUPPORTED` | Statement is factually correct |
| `CONTAINS_UNSUPPORTED_CLAIMS` | Cannot verify (insufficient evidence) |
| `CONTAINS_REFUTED_CLAIMS` | Statement contains false information |

### Confidence Scores
- `0.9 - 1.0`: Very confident
- `0.7 - 0.9`: Confident
- `0.5 - 0.7`: Moderate confidence
- `0.0 - 0.5`: Low confidence

## Common Patterns

### Pattern 1: Simple Verification
```python
def verify_statement(statement: str) -> bool:
    """Return True if statement is supported."""
    adaptive_judge = AdaptiveJudgeModule()
    result = adaptive_judge(statement=statement)
    return result.overall_verdict == "SUPPORTED"
```

### Pattern 2: Detailed Analysis
```python
def analyze_statement(statement: str) -> dict:
    """Get full analysis with research details."""
    adaptive_judge = AdaptiveJudgeModule()
    result = adaptive_judge(statement=statement)

    return {
        "verdict": result.overall_verdict,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "research_performed": result.fallback_triggered,
        "claims_analyzed": len(result.claims) if result.fallback_triggered else 0
    }
```

### Pattern 3: Batch Processing
```python
def batch_verify(statements: list[str]) -> list[dict]:
    """Verify multiple statements."""
    adaptive_judge = AdaptiveJudgeModule()

    results = []
    for statement in statements:
        result = adaptive_judge(statement=statement)
        results.append({
            "statement": statement,
            "verdict": result.overall_verdict,
            "confidence": result.confidence,
            "fallback_used": result.fallback_triggered
        })

    return results
```

### Pattern 4: API Endpoint
```python
from fastapi import FastAPI

app = FastAPI()
adaptive_judge = AdaptiveJudgeModule()

@app.post("/fact-check")
def fact_check(statement: str):
    result = adaptive_judge(statement=statement)
    return {
        "verdict": result.overall_verdict,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "research_performed": result.fallback_triggered
    }
```

## Performance Guide

| Scenario | Time | Cost | When to Use |
|----------|------|------|-------------|
| **Fast Path** (no fallback) | 1-2s | Low | Known facts, confident answers |
| **Slow Path** (with fallback) | 10-30s | High | Uncertain claims, recent events |

### Optimize for Speed
```python
adaptive_judge = AdaptiveJudgeModule(
    confidence_threshold=0.5,  # Less likely to trigger fallback
    enable_fallback=False      # Or disable entirely
)
```

### Optimize for Accuracy
```python
adaptive_judge = AdaptiveJudgeModule(
    confidence_threshold=0.9,  # More likely to trigger fallback
    max_judge_iterations=5,    # More thorough research
    max_page_visits=5
)
```

## Troubleshooting

### Issue: Fallback never triggers
**Possible causes:**
- `enable_fallback=False`
- Threshold too low (e.g., 0.3)
- LLM always returns high confidence

**Solution:**
```python
# Increase threshold
adaptive_judge = AdaptiveJudgeModule(confidence_threshold=0.8)

# Or check verdict types
result = adaptive_judge(statement="...")
print(f"Verdict: {result.overall_verdict}, Confidence: {result.confidence}")
```

### Issue: Fallback triggers too often
**Possible causes:**
- Threshold too high (e.g., 0.95)
- LLM frequently uncertain

**Solution:**
```python
# Lower threshold
adaptive_judge = AdaptiveJudgeModule(confidence_threshold=0.6)

# Monitor fallback rate
fallback_count = sum(r.fallback_triggered for r in results)
rate = fallback_count / len(results)
print(f"Fallback rate: {rate:.1%}")
```

### Issue: Slow performance
**Possible causes:**
- Fallback triggering frequently
- Heavy web research

**Solution:**
```python
# Reduce research depth
adaptive_judge = AdaptiveJudgeModule(
    max_judge_iterations=2,
    max_page_visits=2
)

# Or disable fallback
adaptive_judge = AdaptiveJudgeModule(enable_fallback=False)
```

### Issue: Missing API keys error
**Solution:**
```bash
# Set required keys
export OPENAI_API_KEY="your-key"

# If fallback triggers, also set:
export SERPER_API_KEY="your-key"
export FIRECRAWL_API_KEY="your-key"
```

## Next Steps

1. **Try the examples:**
   ```bash
   python examples/adaptive_judge_example.py
   ```

2. **Read the full documentation:**
   - `src/factchecker/modules/README_ADAPTIVE_JUDGE.md`
   - `ADAPTIVE_JUDGE_SUMMARY.md`

3. **Explore the flowchart:**
   - `src/factchecker/modules/ADAPTIVE_JUDGE_FLOWCHART.txt`

4. **Run tests:**
   ```bash
   python verify_adaptive_judge.py
   ```

## Need Help?

- **Documentation**: See `README_ADAPTIVE_JUDGE.md` for detailed guide
- **Examples**: Check `examples/adaptive_judge_example.py`
- **Tests**: Review `tests/test_adaptive_judge_module.py`
- **Code**: Read `src/factchecker/modules/adaptive_judge_module.py`

## Key Takeaways

✅ **Intelligent Routing**: Automatically decides when web research is needed
✅ **Confidence-Based**: Uses LLM's own uncertainty signal
✅ **Configurable**: Adjust threshold and behavior to your needs
✅ **Efficient**: Fast path for known facts, slow path for uncertain claims
✅ **Transparent**: Always shows whether research was performed

---

**Ready to start? Just 3 lines:**

```python
from src.factchecker.modules import AdaptiveJudgeModule
adaptive_judge = AdaptiveJudgeModule()
result = adaptive_judge(statement="Your statement here")
```
