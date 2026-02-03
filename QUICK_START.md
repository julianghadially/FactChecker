# Quick Start Guide - Adaptive JudgeModule

## Installation
No additional dependencies needed! The enhancement uses existing services in your codebase.

## Basic Usage

```python
import dspy
from src.factchecker.simple.modules.judge_module import JudgeModule

# 1. Configure your LLM
lm = dspy.OpenAI(model="gpt-4")  # or Claude, etc.
dspy.settings.configure(lm=lm)

# 2. Create judge with adaptive search
judge = JudgeModule(enable_adaptive_search=True)

# 3. Check a statement
result = judge.forward("The 2024 Olympics were held in Paris")

# 4. View results
print(f"Verdict: {result.overall_verdict}")
print(f"Confidence: {result.confidence}")
print(f"Web search triggered: {result.web_search_triggered}")
```

## Configuration Presets

### Default (Balanced)
```python
judge = JudgeModule()  # enable_adaptive_search=True by default
```
- Confidence threshold: 0.6
- Search results: 3
- Best for: General purpose fact-checking

### Fast Mode
```python
judge = JudgeModule(
    confidence_threshold=0.4,  # Higher bar = fewer searches
    num_search_results=2
)
```
- Best for: High-volume checking where speed matters
- ~10-15% search trigger rate

### Accurate Mode
```python
judge = JudgeModule(
    confidence_threshold=0.8,  # Lower bar = more searches
    num_search_results=5,
    max_scrape_length=10000
)
```
- Best for: Critical fact-checking, recent events
- ~30-40% search trigger rate

### Original Mode (No Search)
```python
judge = JudgeModule(enable_adaptive_search=False)
```
- Best for: Historical facts, offline use
- 0% search trigger rate (no external calls)

## Understanding Results

### When Search is NOT Triggered
```python
result = judge.forward("World War II ended in 1945")

# Output:
{
    "statement": "World War II ended in 1945",
    "overall_verdict": "SUPPORTED",
    "confidence": 0.95,
    "reasoning": "This is a well-established historical fact...",
    "web_search_triggered": False,
    "evidence": ""
}
```

### When Search IS Triggered
```python
result = judge.forward("A major tech company announced new AI today")

# Output:
{
    "statement": "A major tech company announced new AI today",
    "overall_verdict": "CONTAINS_UNSUPPORTED_CLAIMS",
    "confidence": 0.75,
    "reasoning": "Based on recent web sources...",
    "web_search_triggered": True,
    "evidence": "=== Source 1: TechCrunch ===\nURL: ...\nContent: ...",
    "initial_confidence": 0.45,
    "initial_reasoning": "This is beyond my knowledge cutoff..."
}
```

## Common Use Cases

### Recent Events
```python
# Automatically triggers search for recent claims
statements = [
    "The 2024 US election winner was announced",
    "A new iPhone model was released this week",
    "There was an earthquake in Japan yesterday"
]

for stmt in statements:
    result = judge.forward(stmt)
    if result.web_search_triggered:
        print(f"✓ Found recent evidence for: {stmt}")
```

### Mixed Batch Processing
```python
statements = [
    "The Earth is round",           # No search needed
    "Paris is the capital of France", # No search needed
    "A new planet was discovered last month", # Will search
    "The COVID-19 pandemic started in 2019", # No search needed
]

search_count = 0
for stmt in statements:
    result = judge.forward(stmt)
    if result.web_search_triggered:
        search_count += 1

print(f"Searched for {search_count}/{len(statements)} statements")
# Expected: ~1/4 (25%)
```

### Monitoring Search Triggers
```python
# Track which statements trigger searches for optimization
search_log = []

for statement in your_statements:
    result = judge.forward(statement)
    
    if result.web_search_triggered:
        search_log.append({
            "statement": statement,
            "initial_conf": result.initial_confidence,
            "final_conf": result.confidence,
            "improvement": result.confidence - result.initial_confidence
        })

# Analyze patterns
avg_improvement = sum(s["improvement"] for s in search_log) / len(search_log)
print(f"Average confidence improvement from search: {avg_improvement:.2f}")
```

## Environment Setup

### Required API Keys
```bash
export SERPER_API_KEY="your-serper-key"      # For web search
export FIRECRAWL_API_KEY="your-firecrawl-key" # For web scraping
```

### Getting API Keys
- **Serper**: https://serper.dev (free tier: 2,500 searches/month)
- **Firecrawl**: https://firecrawl.dev (free tier available)

## Testing Your Setup

```python
# Run basic tests
python test_adaptive_judge.py

# Run examples
python example_adaptive_judge_usage.py
```

## Troubleshooting

### "Module works but never triggers search"
- Check API keys are set: `echo $SERPER_API_KEY`
- Lower confidence threshold: `JudgeModule(confidence_threshold=0.8)`
- Test with recent event statement

### "Search triggered but no evidence"
- API keys may be invalid
- Check internet connection
- Review error in `result.evidence` field

### "Too many searches being triggered"
- Raise confidence threshold: `JudgeModule(confidence_threshold=0.4)`
- Review what phrases trigger searches in documentation

## Performance Tips

1. **Batch similar claims**: Process historical facts separately from recent events
2. **Cache results**: Store results for frequently checked statements
3. **Adjust per domain**: Use higher threshold for general knowledge domains
4. **Monitor metrics**: Track search rate and adjust threshold accordingly

## Next Steps

1. ✅ Read `ADAPTIVE_JUDGE_ENHANCEMENT.md` for full documentation
2. ✅ Review `CHANGES_SUMMARY.md` to understand what changed
3. ✅ Run `test_adaptive_judge.py` to verify installation
4. ✅ Try examples in `example_adaptive_judge_usage.py`
5. ✅ Integrate into your fact-checking pipeline

## Support

For issues or questions:
1. Check the comprehensive documentation in `ADAPTIVE_JUDGE_ENHANCEMENT.md`
2. Review test cases in `test_adaptive_judge.py`
3. See usage patterns in `example_adaptive_judge_usage.py`

---

**Ready to enhance your fact-checking with adaptive web search!** 🚀
