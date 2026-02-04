# Temporal Research Router - Quick Start Guide

## What It Does

Automatically detects when a fact-checking claim is about recent events and routes it to Google News search instead of regular web search for more accurate, timely results.

## How to Use

### Option 1: Automatic (Recommended) ✨

Just use the pipeline normally - temporal routing happens automatically!

```python
from src.factchecker.modules import FactCheckerPipeline

pipeline = FactCheckerPipeline()

# Temporal claim - automatically uses news search
result = pipeline(statement="Apple just announced a new iPhone today")

# Non-temporal claim - automatically uses web search
result = pipeline(statement="Paris is the capital of France")
```

**That's it!** The pipeline handles everything.

### Option 2: Direct Router Use

```python
from src.factchecker.modules import TemporalResearchRouterModule

router = TemporalResearchRouterModule(max_page_visits=3)

evidence = router(
    claim="Tesla has opened a new factory",
    query="Tesla factory"
)
```

### Option 3: Manual Control

```python
from src.factchecker.modules import ResearchAgentModule

research = ResearchAgentModule(max_page_visits=3)

# Force news search with daily recency
evidence = research(
    claim="Some claim",
    query="search query",
    use_news_search=True,
    news_recency="d"  # "d" = day, "w" = week, "m" = month
)

# Force regular search
evidence = research(
    claim="Some claim",
    query="search query",
    use_news_search=False
)
```

## What Claims Are Considered "Temporal"?

### ✅ Temporal (Uses News Search)

**Very Recent (past 24 hours)**
- "Apple **just announced** a new iPhone **today**"
- "The court **ruled yesterday** on the case"
- "**Breaking**: New policy announced **this week**"

**Recent (past week)**
- "Microsoft **recently launched** AI features"
- "Tesla **this month** opened a factory"
- "The **latest** unemployment numbers"

**Current/Recent Events (past month)**
- "Amazon **has opened** new warehouses in 2024"
- "The government **upgraded** security systems"
- "Supreme Court **ruled** on abortion rights"
- "The company announced **quarterly earnings**"

### ❌ Non-Temporal (Uses Regular Search)

- "Paris is the capital of France"
- "Water boils at 100°C"
- "Shakespeare wrote Hamlet"
- "The Earth orbits the Sun"

## When Will It Use News vs. Web Search?

| Temporal Signal | Recency | Search Type | Example |
|----------------|---------|-------------|---------|
| "today", "just", "breaking" | Daily ("d") | Google News (24h) | "Apple **just** announced" |
| "recently", "this month" | Weekly ("w") | Google News (7d) | "Microsoft **recently** launched" |
| "has opened", "2024", "ruled" | Monthly ("m") | Google News (30d) | "Tesla **has opened** factory" |
| No temporal signals | N/A | Regular Google | "Paris is the capital" |

## What Gets Enhanced?

The router automatically improves your search queries:

```
Claim: "Apple just announced iPhone today"
Query: "Apple iPhone"
Enhanced: "Apple iPhone February 2024"
→ More specific, better results
```

## Debugging

The router prints what it's doing:

```
[TemporalRouter] Detected temporal claim with recency 'd'
[TemporalRouter] Original query: Apple iPhone
[TemporalRouter] Enriched query: Apple iPhone February 2024
[TemporalRouter] Routing to NEWS search
```

or:

```
[TemporalRouter] No temporal signals detected
[TemporalRouter] Routing to REGULAR web search
```

## Testing Your Claims

Want to see if your claim is detected as temporal? Run the test file:

```bash
python src/factchecker/modules/temporal_router_example.py
```

## Common Patterns Detected

### Legal/Government
- ✅ "court ruled", "judge ordered", "government announced"
- ✅ "approved", "rejected", "mandated", "declared"

### Company News
- ✅ "has opened", "launched", "announced", "released"
- ✅ "IPO", "acquisition", "merger", "bankruptcy"
- ✅ "CEO resigned", "executive appointed"

### Market/Finance
- ✅ "stock", "quarterly earnings", "shares", "trading"
- ✅ "market performance", "financial results"

### Time References
- ✅ "today", "yesterday", "this week/month/year"
- ✅ "recently", "just", "newly", "latest"
- ✅ "January 2024", "2024-01-15", "in 2024"

## Performance

- **Temporal claims**: 🚀 Better results from news sources
- **Non-temporal claims**: ⚡ Same as before (no change)
- **No overhead**: Detection is fast (~1ms per claim)

## Configuration

### Adjust Page Visits

```python
pipeline = FactCheckerPipeline(
    max_page_visits=5  # Visit more pages per search
)
```

### Adjust Iterations

```python
pipeline = FactCheckerPipeline(
    max_judge_iterations=5  # More research iterations per claim
)
```

## Troubleshooting

### "My temporal claim isn't being detected"

Check if your claim contains temporal signals:
- Action verbs: "opened", "announced", "ruled"
- Time words: "today", "recently", "this year"
- Dates: "2024", "January 15", etc.

### "My claim should use regular search, not news"

The router might be too sensitive. Consider:
- Rephrasing without temporal words
- Or use `ResearchAgentModule` directly with `use_news_search=False`

### "Not getting good results"

Try:
1. Check the query enrichment in debug output
2. Verify the recency level is appropriate
3. Consider manual routing if needed

## More Information

- 📖 Full documentation: `TEMPORAL_ROUTER_README.md`
- 🔄 Flow diagrams: `TEMPORAL_ROUTER_FLOW.md`
- 📝 Implementation details: `TEMPORAL_ROUTER_IMPLEMENTATION_SUMMARY.md`
- 🧪 Test examples: `temporal_router_example.py`

## Questions?

The temporal router is designed to "just work" - you don't need to think about it. If you're using `FactCheckerPipeline`, you're already using it! 🎉
