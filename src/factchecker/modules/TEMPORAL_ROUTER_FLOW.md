# Temporal Research Router - System Flow

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      FactCheckerPipeline                         │
│  (Orchestrates the complete fact-checking process)              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ClaimExtractorModule                          │
│  Extracts individual claims from the statement                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   For each claim...   │
                 └───────────┬───────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FireJudgeModule                             │
│  Iteratively evaluates claims with web research                 │
│  (max_iterations = 3 by default)                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
           ┌─────────────────────────────────────┐
           │  Need more evidence? Generate query │
           └─────────────────┬───────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              TemporalResearchRouterModule (NEW)                  │
│  Intelligently routes to news or web search                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
          ┌──────────────────────────────────────┐
          │   Temporal Signal Detection          │
          │                                      │
          │   1. Analyze claim text             │
          │   2. Detect temporal patterns       │
          │   3. Classify recency level         │
          └──────────────┬───────────────────────┘
                         │
         ┌───────────────┴────────────────┐
         │                                │
    [Temporal?]                    [Non-temporal]
         │                                │
         ▼                                ▼
    ┌────────┐                   ┌───────────────┐
    │  YES   │                   │      NO       │
    └───┬────┘                   └───────┬───────┘
        │                                │
        ▼                                │
┌──────────────────┐                     │
│ Recency Level?   │                     │
├──────────────────┤                     │
│ • Daily ("d")    │                     │
│ • Weekly ("w")   │                     │
│ • Monthly ("m")  │                     │
└─────────┬────────┘                     │
          │                              │
          ▼                              │
┌──────────────────┐                     │
│ Enrich Query     │                     │
│ • Add year/month │                     │
│ • Add date info  │                     │
└─────────┬────────┘                     │
          │                              │
          ▼                              │
┌─────────────────────────────────────────────────────────────────┐
│                    ResearchAgentModule                           │
│  (Modified to accept news search parameters)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┴───────────────────┐
         │                                       │
         ▼                                       ▼
┌─────────────────────┐              ┌──────────────────────┐
│  SerperService      │              │  SerperService       │
│  .search_news()     │              │  .search()           │
│                     │              │                      │
│  • Google News      │              │  • Regular Google    │
│  • With recency:    │              │  • All types         │
│    - "d" (day)      │              │  • Broad coverage    │
│    - "w" (week)     │              │                      │
│    - "m" (month)    │              │                      │
└──────────┬──────────┘              └──────────┬───────────┘
           │                                    │
           └────────────────┬───────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │   Search Results        │
              │   (Top 10 results)      │
              └─────────┬───────────────┘
                        │
                        ▼
              ┌─────────────────────────┐
              │   PageSelector (LLM)    │
              │   Selects best page     │
              └─────────┬───────────────┘
                        │
                        ▼
              ┌─────────────────────────┐
              │   FirecrawlService      │
              │   Scrapes page content  │
              └─────────┬───────────────┘
                        │
                        ▼
              ┌─────────────────────────┐
              │  EvidenceSummarizer     │
              │  Extracts evidence      │
              └─────────┬───────────────┘
                        │
                        ▼
          ┌─────────────────────────────┐
          │  Repeat for max_page_visits │
          │  (default: 3 pages)         │
          └─────────────┬───────────────┘
                        │
                        ▼
              ┌─────────────────────────┐
              │  Aggregated Evidence    │
              └─────────┬───────────────┘
                        │
                        │ (back to FireJudgeModule)
                        ▼
              ┌─────────────────────────┐
              │  Judge makes verdict    │
              │  or requests more       │
              │  research               │
              └─────────┬───────────────┘
                        │
                        ▼
              ┌─────────────────────────┐
              │  All claims evaluated   │
              └─────────┬───────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AggregatorModule                              │
│  Combines all claim verdicts into overall statement verdict     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Final Result    │
                    │  • Overall       │
                    │    verdict       │
                    │  • Confidence    │
                    │  • Reasoning     │
                    │  • All claims    │
                    └──────────────────┘
```

## Temporal Detection Logic Detail

```
┌─────────────────────────────────────────────────────────────────┐
│                  Temporal Signal Detection                       │
└─────────────────────────────────────────────────────────────────┘

Input: "Apple just announced a new iPhone today"
       │
       ▼
┌──────────────────────────────────────┐
│  Pattern Matching                    │
│  • VERY_RECENT_PHRASES               │
│    ✓ "just"                          │
│    ✓ "today"                         │
│  • TEMPORAL_PHRASES                  │
│    ✓ "announced"                     │
└──────────────┬───────────────────────┘
               │
               ▼
        ┌──────────────┐
        │  Match found │
        └──────┬───────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Classify Recency                    │
│  "just" + "today" → VERY RECENT      │
│  Recency Level: "d" (daily)          │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Enrich Query                        │
│  Query: "Apple iPhone"               │
│  + Current: "February 2024"          │
│  → "Apple iPhone February 2024"      │
└──────────────┬───────────────────────┘
               │
               ▼
        ┌──────────────┐
        │  Route to    │
        │  News Search │
        │  (past day)  │
        └──────────────┘
```

## Example Routing Decisions

### Example 1: Very Recent Event
```
Claim: "The Supreme Court ruled today on student loans"

Detection:
  • Pattern: "ruled" (TEMPORAL_PHRASES)
  • Pattern: "today" (VERY_RECENT_PHRASES)
  → is_temporal = True
  → recency = "d" (daily)

Query Enrichment:
  • Original: "Supreme Court student loans"
  • Enriched: "Supreme Court student loans February 2024"

Routing:
  → SerperService.search_news(query, recency="d")
  → Search: Google News (past 24 hours)
```

### Example 2: Recent Announcement
```
Claim: "Tesla recently launched Autopilot updates"

Detection:
  • Pattern: "recently launched" (RECENT_PHRASES)
  → is_temporal = True
  → recency = "w" (weekly)

Query Enrichment:
  • Original: "Tesla Autopilot updates"
  • Enriched: "Tesla Autopilot updates 2024"

Routing:
  → SerperService.search_news(query, recency="w")
  → Search: Google News (past week)
```

### Example 3: Current Year Event
```
Claim: "Microsoft upgraded Azure services in 2024"

Detection:
  • Pattern: "upgraded" (TEMPORAL_PHRASES)
  • Pattern: "in 2024" (DATE_PATTERNS)
  → is_temporal = True
  → recency = "m" (monthly)

Query Enrichment:
  • Original: "Microsoft Azure services"
  • Already has year, no enrichment needed

Routing:
  → SerperService.search_news(query, recency="m")
  → Search: Google News (past month)
```

### Example 4: Non-Temporal Fact
```
Claim: "Paris is the capital of France"

Detection:
  • No temporal patterns detected
  → is_temporal = False
  → recency = ""

Query Enrichment:
  • No enrichment needed

Routing:
  → SerperService.search(query)
  → Search: Regular Google search (all time)
```

## Key Benefits

1. **Automatic Detection**: No manual configuration needed per claim
2. **Optimized Search**: Uses appropriate search type and recency
3. **Better Results**: News search for temporal claims gives fresher, more relevant results
4. **Contextual Enrichment**: Queries are enhanced with temporal context
5. **Seamless Integration**: Works transparently within existing pipeline
6. **Fallback Safety**: Non-temporal claims use regular search as before

## Performance Impact

- **Temporal claims**: 🚀 Faster, more accurate results from news sources
- **Non-temporal claims**: ✓ No performance change, same as before
- **Mixed statements**: 🎯 Each claim routed optimally
