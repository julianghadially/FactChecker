# Testing the Enhanced JudgeModule

## Quick Start

### Prerequisites

1. **Environment Variables**
   ```bash
   export OPENAI_AGENTJUDGEJG_KEY="your-openai-key"
   export SERPER_KEY="your-serper-key"
   ```

2. **Dependencies**
   ```bash
   pip install dspy requests firecrawl-py
   ```

### Run the Main Test

```bash
python test_judge_with_search.py
```

This will test 5 different scenarios and show how the enhanced module handles them.

### Run Usage Examples

```bash
python examples/simple_judge_usage.py
```

This demonstrates 5 practical usage patterns.

## Test Scenarios

### Scenario 1: Historical Fact
**Statement**: "The Great Wall of China was built over several centuries..."

**Expected Behavior**:
- ✅ No web search triggered (LLM has knowledge)
- ✅ High confidence verdict
- ✅ Fast response (~1.5s)

**Purpose**: Verify no unnecessary searches on historical facts

---

### Scenario 2: Recent Event (OpenAI)
**Statement**: "OpenAI released GPT-5 in early 2025..."

**Expected Behavior**:
- ✅ Web search triggered (beyond training data)
- ✅ Search results examined
- ✅ Verdict based on web evidence
- ⏱️ Medium response (~2.5s)

**Purpose**: Verify web search for recent tech events

---

### Scenario 3: Recent Event (Election)
**Statement**: "Donald Trump won the 2024 U.S. presidential election."

**Expected Behavior**:
- ✅ Web search triggered
- ✅ Multiple news sources consulted
- ✅ Definitive verdict (SUPPORTED or REFUTED)
- ⏱️ Medium response (~2.5s)

**Purpose**: Verify web search for verifiable recent events

---

### Scenario 4: Clear False Statement
**Statement**: "The Earth is flat and orbits around the Moon."

**Expected Behavior**:
- ✅ No web search triggered (LLM knows this is false)
- ✅ CONTAINS_REFUTED_CLAIMS verdict
- ✅ High confidence
- ✅ Fast response (~1.5s)

**Purpose**: Verify no unnecessary searches on obvious falsehoods

---

### Scenario 5: Uncertain Future Event
**Statement**: "SpaceX launched its first crewed mission to Mars in 2025."

**Expected Behavior**:
- ✅ Web search triggered (needs verification)
- ✅ Search results examined
- ✅ Verdict based on available evidence
- ⏱️ Medium response (~2.5s)

**Purpose**: Verify handling of uncertain future/recent claims

## Expected Output

### Test with Web Search Enabled

```
================================================================================
Statement 1: The Great Wall of China was built over several centuries...
--------------------------------------------------------------------------------
Verdict: SUPPORTED
Confidence: 0.95
Web Search Performed: False
Reasoning: This is a well-established historical fact...
```

### Test with Web Search Disabled

```
================================================================================
Statement: Donald Trump won the 2024 U.S. presidential election.
--------------------------------------------------------------------------------
Verdict: CONTAINS_UNSUPPORTED_CLAIMS
Confidence: 0.50
Web Search Performed: False
Reasoning: Cannot verify due to knowledge cutoff...
```

## Interpreting Results

### Success Indicators

✅ **Historical facts**: No search, high confidence, correct verdict
✅ **Recent events**: Search triggered, search results used, definitive verdict
✅ **False claims**: No search, high confidence, REFUTED verdict
✅ **Response times**: ~1.5s without search, ~2.5s with search

### Potential Issues

⚠️ **Search always triggering**: Check detection patterns
⚠️ **Search never triggering**: Verify enable_web_search=True
⚠️ **Search failing**: Check SERPER_KEY, API limits, connectivity
⚠️ **Wrong verdicts**: Review search results quality, consider query tuning

## Debugging

### Enable Verbose Output

Modify the test script to print intermediate results:

```python
# In test_judge_with_search.py
result = judge(statement=statement)

# Add debugging
print(f"Initial reasoning: {result.reasoning}")
if hasattr(result, 'search_results'):
    print(f"Search results: {result.search_results}")
```

### Check Detection Logic

```python
from src.factchecker.simple.modules.judge_module import JudgeModule

judge = JudgeModule(enable_web_search=True)

# Test detection directly
reasoning = "I cannot verify this due to my knowledge cutoff..."
verdict = "CONTAINS_UNSUPPORTED_CLAIMS"

detected = judge._detect_knowledge_limitations(reasoning, verdict)
print(f"Would trigger search: {detected}")
```

### Test Search Independently

```python
from src.services.serper_service import SerperService

serper = SerperService()
results = serper.search("Donald Trump 2024 election", num_results=5)

for result in results:
    print(f"{result.title}: {result.snippet}")
```

## Performance Benchmarks

Run the test suite and measure:

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Historical facts accuracy | >95% | Count correct verdicts |
| Recent events accuracy | >90% | Count correct verdicts |
| Search trigger rate | 20-30% | Count web_search_performed=True |
| Avg response time (no search) | <2s | Time statements without search |
| Avg response time (with search) | <3s | Time statements with search |

## Validation Checklist

Before considering the enhancement production-ready:

- [ ] All 5 test scenarios pass
- [ ] Historical facts don't trigger search
- [ ] Recent events trigger search appropriately
- [ ] Search results are properly formatted
- [ ] Verdicts improve with search context
- [ ] Response times are acceptable
- [ ] API costs are reasonable
- [ ] Error handling works (disconnect API keys and test)
- [ ] Backward compatibility maintained (test with enable_web_search=False)

## Cost Monitoring

Track API usage:

```python
import time

judge = JudgeModule(enable_web_search=True)
statements = [...]  # Your test statements

search_count = 0
total_time = 0

for stmt in statements:
    start = time.time()
    result = judge(statement=stmt)
    elapsed = time.time() - start

    total_time += elapsed
    if result.web_search_performed:
        search_count += 1

print(f"Searches: {search_count}/{len(statements)} ({search_count/len(statements):.1%})")
print(f"Avg time: {total_time/len(statements):.2f}s")
print(f"Est. cost: ${search_count * 0.001:.3f} (Serper)")
```

## Integration Testing

Test with the full evaluation system:

```python
from src.evaluation.evaluate import run_evaluation
from src.factchecker.simple.modules.judge_module import JudgeModule
from src.baseline.baseline_model import BaselineModel

# Configure DSPy
import dspy
from src.context_.context import openai_key
dspy.configure(lm=dspy.LM("openai/gpt-5-mini", api_key=openai_key))

# Create enhanced judge
judge = JudgeModule(enable_web_search=True)
baseline = BaselineModel()

# Run evaluation on a small sample
results = run_evaluation(
    fact_checker=judge,
    baseline_model=baseline,
    sample_size=10,
    dataset_path="data/FactChecker_news_claims.csv"
)

# Check metrics
print(f"Accuracy: {results['factchecker']['accuracy']:.1%}")
print(f"Error count: {results['factchecker']['error_count']}")
```

## Troubleshooting Guide

### Problem: "ImportError: No module named 'src'"

**Solution**: Run from project root:
```bash
cd /path/to/workspace
python test_judge_with_search.py
```

### Problem: "SerperService: 401 Unauthorized"

**Solution**: Check API key:
```bash
echo $SERPER_KEY
# If empty, set it:
export SERPER_KEY="your-key-here"
```

### Problem: "Search results empty"

**Possible Causes**:
1. API rate limit reached
2. Invalid search query
3. Network connectivity issues

**Solution**: Check Serper dashboard, test connectivity

### Problem: "Search triggered too often/rarely"

**Solution**: Adjust detection patterns in `judge_module.py`:

```python
# In _detect_knowledge_limitations method
# Add/remove patterns as needed
limitation_patterns = [
    r"knowledge cutoff",
    # ... add more patterns
]
```

### Problem: "Wrong verdicts even with search"

**Possible Causes**:
1. Search results not relevant
2. Query not specific enough
3. Contradictory search results

**Solution**:
- Review search results manually
- Consider implementing query extraction with LLM
- Add result ranking/filtering

## Next Steps

After successful testing:

1. **Run Full Evaluation**
   ```bash
   python -m src.main --mode evaluate --sample-size 100
   ```

2. **Compare with Baseline**
   - Check accuracy improvements
   - Measure cost increase
   - Assess response time impact

3. **Optimize if Needed**
   - Tune detection patterns
   - Adjust search result count
   - Implement caching

4. **Deploy to Production**
   - Update default behavior
   - Monitor performance
   - Track costs

## Support

For issues or questions:
1. Check `JUDGE_MODULE_ENHANCEMENT.md` for detailed documentation
2. Review `BEFORE_AFTER_COMPARISON.md` for expected behavior
3. See `CHANGES_SUMMARY.md` for complete change list

## Summary

The test suite validates that the enhanced JudgeModule:
- ✅ Maintains accuracy on historical facts
- ✅ Improves accuracy on recent events
- ✅ Triggers search selectively (efficient)
- ✅ Handles errors gracefully
- ✅ Provides backward compatibility

**Run the tests, verify the results, and you're ready to use the enhanced JudgeModule!**
