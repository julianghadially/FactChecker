# Multi-Query Search Enhancement - Flow Diagram

## Overall Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        JudgeModule.forward()                     │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    First Pass:       │
                    │  Judge (LLM only)    │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Should trigger       │
                    │   research?          │
                    └──────────────────────┘
                          /        \
                       NO/          \YES
                        /            \
                       ▼              ▼
            ┌──────────────┐  ┌──────────────────┐
            │ Return first │  │ _gather_evidence │
            │ pass result  │  │   (enhanced!)    │
            └──────────────┘  └──────────────────┘
                                      │
                                      ▼
                          ┌──────────────────────┐
                          │   Second Pass:       │
                          │ Judge (with evidence)│
                          └──────────────────────┘
                                      │
                                      ▼
                          ┌──────────────────────┐
                          │ Return final result  │
                          └──────────────────────┘
```

## Enhanced _gather_evidence() Method Flow

### Before Enhancement (Single Query)

```
┌─────────────────────────────────────────────────────────┐
│              _gather_evidence(statement)                │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ Search with full statement    │
        │ (1 query, 2 results)          │
        └───────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ Scrape top 2 URLs             │
        └───────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ Return formatted evidence     │
        └───────────────────────────────┘

Problems:
❌ Single broad query may miss specific evidence
❌ Only 2 sources = limited coverage
❌ No targeting of temporal/numeric claims
```

### After Enhancement (Multi-Query)

```
┌──────────────────────────────────────────────────────────────┐
│              _gather_evidence(statement)                     │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────────┐
        │  Step 1: QueryGenerator             │
        │  Generate 1-3 focused queries       │
        │                                     │
        │  Example input:                     │
        │  "Mondelez has been selling         │
        │   sugar-free Oreo cookies..."       │
        │                                     │
        │  Example output:                    │
        │  1. "Oreo Zero Sugar launch date"   │
        │  2. "sugar-free Oreo US history"    │
        │  3. "Mondelez sugar-free products"  │
        └────────────────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────────┐
        │  Step 2: Execute All Queries        │
        │                                     │
        │  For each query:                    │
        │    • Search (3 results per query)   │
        │    • Deduplicate by URL             │
        │    • Collect up to 4 total sources  │
        │                                     │
        │  Query 1 → [URL1, URL2, URL3]       │
        │  Query 2 → [URL4, URL2*]  *dup      │
        │  Query 3 → [URL5, URL1*]  *dup      │
        │                                     │
        │  Final: [URL1, URL2, URL3, URL4]    │
        └────────────────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────────┐
        │  Step 3: Scrape Sources             │
        │  • Scrape top 3-4 URLs              │
        │  • Max 5000 chars per source        │
        │  • Format with title and URL        │
        └────────────────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────────┐
        │  Return Aggregated Evidence         │
        │  "Source 1: ... \n---\n             │
        │   Source 2: ... \n---\n             │
        │   Source 3: ..."                    │
        └────────────────────────────────────┘

Benefits:
✓ Multiple targeted queries find specific evidence
✓ 3-4 sources = better coverage
✓ Deduplication = diverse perspectives
✓ Targets temporal/numeric claims
```

## Example: Oreo Statement Processing

### Input Statement
```
"Mondelez has been selling sugar-free Oreo cookies in the United States
for several years prior to the announced Oreo Zero Sugar launch"
```

### Query Generation (QueryGenerator)
```
┌─────────────────────────────────────────────────────────┐
│  QueryGenerator analyzes statement and identifies:      │
│                                                          │
│  • Key entities: "Mondelez", "Oreo", "United States"    │
│  • Temporal claim: "several years prior"                │
│  • Product names: "sugar-free Oreo", "Oreo Zero Sugar"  │
│  • Verification points:                                 │
│    1. When did Oreo Zero Sugar launch?                  │
│    2. When did sugar-free Oreos appear in US?           │
│    3. What's the timeline relationship?                 │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│  Generated Queries:                                       │
│  1. "Oreo Zero Sugar launch date"                        │
│  2. "sugar-free Oreo United States history before 2026"  │
│  3. "Mondelez sugar-free Oreo products US availability"  │
└──────────────────────────────────────────────────────────┘
```

### Multi-Query Search Execution
```
Query 1: "Oreo Zero Sugar launch date"
┌─────────────────────────────────────────┐
│ Results:                                 │
│ • URL1: mondelez.com/press-release-2026 │
│ • URL2: foodbusinessnews.com/oreo-zero  │
│ • URL3: snackandbakery.com/oreo-launch  │
└─────────────────────────────────────────┘

Query 2: "sugar-free Oreo United States history before 2026"
┌─────────────────────────────────────────┐
│ Results:                                 │
│ • URL4: oreo.com/product-history        │
│ • URL2: foodbusinessnews.com... (dup!)  │
│ • URL5: reddit.com/oreo-discussion      │
└─────────────────────────────────────────┘

Query 3: "Mondelez sugar-free Oreo products US availability"
┌─────────────────────────────────────────┐
│ Results:                                 │
│ • URL1: mondelez.com... (dup!)          │
│ • URL6: walmart.com/oreo-products       │
│ • URL7: target.com/oreo-zero            │
└─────────────────────────────────────────┘

Deduplication → Final URLs: [URL1, URL2, URL3, URL4]
(stops at 4 sources)
```

### Evidence Aggregation
```
┌────────────────────────────────────────────────────────┐
│  Source 1: Mondelez Announces Oreo Zero Sugar Launch   │
│  (mondelez.com/press-release-2026)                     │
│  [Scraped content about January 2026 launch...]        │
│                                                         │
│  ---                                                    │
│                                                         │
│  Source 2: Oreo Zero Sugar Hits Shelves                │
│  (foodbusinessnews.com/oreo-zero)                      │
│  [Scraped content confirming 2026 launch...]           │
│                                                         │
│  ---                                                    │
│                                                         │
│  Source 3: New Oreo Variant Launched                   │
│  (snackandbakery.com/oreo-launch)                      │
│  [Scraped content about product details...]            │
│                                                         │
│  ---                                                    │
│                                                         │
│  Source 4: Oreo Product History                        │
│  (oreo.com/product-history)                            │
│  [Scraped content showing NO sugar-free before 2026]   │
└────────────────────────────────────────────────────────┘
```

### Judge with Evidence
```
Input:
  • Statement: "Mondelez has been selling sugar-free Oreo cookies..."
  • Evidence: [All 4 sources aggregated]

Analysis:
  ✓ Source 1 confirms: Oreo Zero Sugar launched Jan 2026
  ✓ Source 4 confirms: No sugar-free Oreos sold before 2026
  ✗ Statement claims: "several years prior" to launch

Output:
  • Verdict: REFUTED
  • Confidence: 0.95
  • Reasoning: "Evidence clearly shows Oreo Zero Sugar launched
               in January 2026 and there were no sugar-free Oreo
               products sold in the US prior to this launch."
```

## Key Differences Summary

| Aspect | Before | After |
|--------|--------|-------|
| Queries | 1 broad query | 1-3 focused queries |
| Sources | 2 sources | 3-4 sources |
| Targeting | Generic | Temporal/numeric focused |
| Deduplication | No | Yes (by URL) |
| Query Strategy | Use full statement | Extract key search terms |
| Result Quality | Lower accuracy | Higher accuracy |
| Verdict Precision | Often UNSUPPORTED | Can return REFUTED |

## Code Flow Comparison

### Before
```python
def _gather_evidence(statement):
    results = search(statement, num=2)
    evidence = scrape(results[:2])
    return evidence
```

### After
```python
def _gather_evidence(statement):
    queries = query_generator(statement).queries[:3]
    all_results = []
    seen_urls = set()

    for query in queries:
        results = search(query, num=3)
        for result in results:
            if result.url not in seen_urls:
                all_results.append(result)
                seen_urls.add(result.url)
                if len(all_results) >= 4:
                    break

    evidence = scrape(all_results[:4])
    return evidence
```
