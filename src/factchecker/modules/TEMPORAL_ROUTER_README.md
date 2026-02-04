# Temporal Research Router Module

## Overview

The `TemporalResearchRouterModule` is an intelligent routing layer that wraps `ResearchAgentModule` to automatically detect temporal signals in claims and route them to appropriate search methods (news search vs. regular web search) with optimized recency filters.

## Architecture

```
FactCheckerPipeline
    ↓
TemporalResearchRouterModule (new)
    ↓
ResearchAgentModule (modified)
    ↓
SerperService (news_search or regular search)
```

## Key Features

### 1. Temporal Signal Detection

The router analyzes claims for temporal indicators and classifies them into recency levels:

- **Daily (`"d"`)**: Very recent events
  - Examples: "today", "yesterday", "this week", "just", "breaking"
  - Use case: Breaking news, same-day events

- **Weekly (`"w"`)**: Recent events
  - Examples: "this month", "last week", "recently", "newly", "latest"
  - Use case: Recent announcements, weekly news

- **Monthly (`"m"`)**: Temporal but not immediate
  - Examples: "has opened", "upgraded", "ruled", "in 2024"
  - Use case: Recent past events, current year events

### 2. Temporal Phrase Patterns

The module detects various temporal patterns:

#### Recent Actions/Events
- `has/have + [action verb]`: "has opened", "have announced"
- `recently/just/newly + [action]`: "recently launched", "just released"

#### Legal/Governance Actions
- Court actions: "ruled", "declared", "ordered"
- Government actions: "announced", "approved", "mandated"

#### Market/Company News
- Corporate events: "IPO", "acquisition", "merger", "bankruptcy"
- Executive changes: "CEO announced", "executive resigned"
- Financial indicators: "stock", "quarterly earnings", "trading"

#### Date References
- Specific dates: "January 15, 2024", "2024-01-15"
- Relative dates: "this year", "last month", "in 2024"

### 3. Query Enrichment

The router automatically enhances search queries with temporal context:

```python
# Example enrichments:
"Apple iPhone" + claim: "Apple just announced iPhone today"
  → "Apple iPhone February 2024"

"Tesla stock" + claim: "Tesla stock performance this year"
  → "Tesla stock 2024"

"Supreme Court ruling" + claim: "Court ruled on January 15, 2024"
  → "Supreme Court ruling 2024"
```

### 4. Intelligent Routing

```
Claim Analysis
    ↓
Is Temporal? ────No────→ Regular Web Search
    ↓
   Yes
    ↓
Recency Level?
    ├─ Daily (d) ────→ News Search (past 24 hours)
    ├─ Weekly (w) ───→ News Search (past week)
    └─ Monthly (m) ──→ News Search (past month)
```

## Usage

### Basic Usage (Automatic via Pipeline)

The router is automatically used when you run the fact-checking pipeline:

```python
from src.factchecker.modules import FactCheckerPipeline

pipeline = FactCheckerPipeline(
    max_judge_iterations=3,
    max_page_visits=3
)

# The temporal router is automatically used internally
result = pipeline(statement="Apple just announced a new iPhone today")
```

### Direct Usage

You can also use the router directly:

```python
from src.factchecker.modules import TemporalResearchRouterModule

router = TemporalResearchRouterModule(max_page_visits=3)

# Will automatically detect temporal signals and route to news search
evidence = router(
    claim="Tesla has opened a new factory in 2024",
    query="Tesla factory 2024"
)
```

## Implementation Details

### Modified Files

1. **`temporal_research_router_module.py`** (NEW)
   - Main router implementation
   - Temporal signal detection logic
   - Query enrichment logic

2. **`research_agent_module.py`** (MODIFIED)
   - Added `use_news_search` parameter to `forward()`
   - Added `news_recency` parameter to `forward()`
   - Added logic to convert news articles to SearchResult format

3. **`fact_checker_pipeline.py`** (MODIFIED)
   - Changed from `ResearchAgentModule` to `TemporalResearchRouterModule`
   - Passes router to `FireJudgeModule`

4. **`fire_judge_module.py`** (MODIFIED)
   - Updated type hints to accept both `ResearchAgentModule` and `TemporalResearchRouterModule`
   - No behavioral changes needed (duck typing)

## Configuration

### Router Parameters

```python
TemporalResearchRouterModule(
    max_page_visits=3  # Passed to underlying ResearchAgentModule
)
```

### Detection Tuning

To customize temporal detection, modify the class constants in `temporal_research_router_module.py`:

```python
class TemporalResearchRouterModule(dspy.Module):
    # Add new temporal phrases
    TEMPORAL_PHRASES = [
        r'\b(has|have)\s+(opened|launched|...)\b',
        # Add your patterns here
    ]

    # Add new very recent indicators
    VERY_RECENT_PHRASES = [
        r'\b(today|yesterday|...)\b',
        # Add your patterns here
    ]
```

## Examples

### Example 1: Breaking News
```python
claim = "The Supreme Court ruled today on student loan forgiveness"
# Detected: temporal=True, recency="d"
# Query: "Supreme Court student loan" → "Supreme Court student loan February 2024"
# Search: Google News (past 24 hours)
```

### Example 2: Recent Announcement
```python
claim = "Microsoft recently upgraded their Azure AI services"
# Detected: temporal=True, recency="w"
# Query: "Microsoft Azure AI" → "Microsoft Azure AI 2024"
# Search: Google News (past week)
```

### Example 3: Current Year Event
```python
claim = "Tesla has opened a new Gigafactory in 2024"
# Detected: temporal=True, recency="m"
# Query: "Tesla Gigafactory" → "Tesla Gigafactory 2024"
# Search: Google News (past month)
```

### Example 4: Non-Temporal Claim
```python
claim = "Paris is the capital of France"
# Detected: temporal=False, recency=""
# Query: "Paris capital France" (unchanged)
# Search: Regular web search
```

## Testing

Run the example test file to see the router in action:

```bash
python src/factchecker/modules/temporal_router_example.py
```

This will test:
- Temporal signal detection for various claim types
- Query enrichment with temporal context
- Classification into different recency levels

## Performance Considerations

### When Temporal Routing Helps
- ✅ Recent events (within current year)
- ✅ Breaking news claims
- ✅ Company announcements
- ✅ Government/legal actions
- ✅ Market/financial claims

### When Regular Search Is Better
- ✅ Historical facts
- ✅ Scientific facts
- ✅ Geographic information
- ✅ General knowledge
- ✅ Definitions and concepts

## Debugging

The router includes debug print statements:

```
[TemporalRouter] Detected temporal claim with recency 'd'
[TemporalRouter] Original query: Apple iPhone
[TemporalRouter] Enriched query: Apple iPhone February 2024
[TemporalRouter] Routing to NEWS search
```

To disable these, comment out the print statements in the `forward()` method.

## Future Enhancements

Potential improvements:

1. **ML-based Detection**: Train a classifier for temporal signal detection
2. **Dynamic Recency**: Adjust recency based on claim topic (e.g., tech news vs. politics)
3. **Hybrid Search**: Combine news and web results for certain claim types
4. **Temporal Confidence**: Return confidence scores for temporal detection
5. **Geographic Awareness**: Add location-based temporal routing
6. **Language Support**: Extend temporal patterns to other languages

## API Reference

### TemporalResearchRouterModule

#### `__init__(max_page_visits: int = 3)`
Initialize the router with underlying research agent.

#### `forward(claim: str, query: str) -> dspy.Prediction`
Route research request based on temporal signal detection.

**Parameters:**
- `claim`: The claim being fact-checked
- `query`: Search query to execute

**Returns:**
- Evidence from research (same format as ResearchAgentModule)

#### `_detect_temporal_signals(claim: str) -> Tuple[bool, str]`
Detect temporal signals in a claim.

**Returns:**
- Tuple of (has_temporal_signals, recency_filter)
- recency_filter: "d" (day), "w" (week), "m" (month), or "" (not temporal)

#### `_enrich_query_with_temporal_context(query: str, claim: str, recency: str) -> str`
Enrich search query with temporal context.

**Returns:**
- Enhanced query string with year/month/date context

## License

Part of the fact-checker system. See main project license.
