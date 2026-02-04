# Temporal Router Decision Flowchart

## High-Level Flow

```
START: Fact-Check Request
        |
        v
┌───────────────────────┐
│ Extract Information   │
│ • URLs from text      │
│ • Dates from text     │
│ • Temporal keywords   │
└───────┬───────────────┘
        |
        v
┌───────────────────────┐
│  Decision Rules       │
└───────┬───────────────┘
        |
        v
    [Decision Tree]
        |
    ┌───┴───┐
    |       |
    v       v
  Fast    Web
  Path   Research
```

## Detailed Decision Tree

```
                    Input Statement + URLs
                            |
                            v
                   ┌────────────────┐
                   │ URLs Provided? │
                   └────┬───────┬───┘
                        │       │
                   YES  │       │ NO
                        │       │
                        v       v
                    ┌────┐   ┌─────────────────┐
                    │ WEB│   │ Extract Dates   │
                    │ RE-│   │ from Statement  │
                    │SEAR│   └─────┬───────────┘
                    │ CH │         │
                    └────┘         v
                        ▲    ┌──────────────────┐
                        │    │ Dates >= June    │
                        │    │ 2024 (cutoff)?   │
                        │    └─────┬────────┬───┘
                        │          │        │
                        │     YES  │        │ NO
                        │          │        │
                        │          v        v
                        │      ┌────┐   ┌────────────────┐
                        └──────┤ WEB│   │ Has Temporal   │
                               │ RE-│   │ Keywords?      │
                               │SEAR│   └─────┬──────┬───┘
                               │ CH │         │      │
                               └────┘    YES  │      │ NO
                                   ▲          │      │
                                   │          v      v
                                   │      ┌────┐  ┌──────┐
                                   └──────┤ WEB│  │ FAST │
                                          │ RE-│  │ JUDGE│
                                          │SEAR│  └──────┘
                                          │ CH │
                                          └────┘
```

## Rule-Based Logic

```python
# Pseudo-code for routing decision

def should_use_web_research(statement, urls, dates):

    # RULE 1: URLs provided
    if urls:
        return True, "URLs provided"

    # RULE 2: Dates beyond cutoff
    for date in dates:
        if date >= KNOWLEDGE_CUTOFF:  # June 2024
            return True, f"Date beyond cutoff: {date}"

    # RULE 3: Temporal keywords
    if has_temporal_keywords(statement):
        return True, "Temporal keywords found"

    # DEFAULT: Use fast judge
    return False, "No web research needed"
```

## Example Routing Decisions

### Example 1: Historical Statement

```
Input: "The Apollo 11 mission landed on the moon on July 20, 1969."

┌─────────────────────────────────────┐
│ Step 1: Check URLs                  │
│ • URLs found: 0                     │
│ • Decision: Continue                │
└──────────┬──────────────────────────┘
           v
┌─────────────────────────────────────┐
│ Step 2: Extract Dates               │
│ • Dates found: [1969-07-20]         │
│ • Compare: 1969-07-20 < 2024-06-01 │
│ • Decision: Continue                │
└──────────┬──────────────────────────┘
           v
┌─────────────────────────────────────┐
│ Step 3: Check Keywords              │
│ • Keywords found: None              │
│ • Decision: Use JudgeModule         │
└──────────┬──────────────────────────┘
           v
      ┌─────────────┐
      │ JudgeModule │ ✅
      │ (Fast Path) │
      └─────────────┘

Result: "No temporal references or URLs requiring web research"
```

### Example 2: Recent Event

```
Input: "In January 2025, tech companies announced major layoffs."

┌─────────────────────────────────────┐
│ Step 1: Check URLs                  │
│ • URLs found: 0                     │
│ • Decision: Continue                │
└──────────┬──────────────────────────┘
           v
┌─────────────────────────────────────┐
│ Step 2: Extract Dates               │
│ • Dates found: [2025-01-01]         │
│ • Compare: 2025-01-01 >= 2024-06-01│
│ • Decision: Use Web Research        │
└──────────┬──────────────────────────┘
           v
    ┌──────────────────────┐
    │ FactCheckerPipeline  │ ✅
    │ (Web Research)       │
    └──────────────────────┘

Result: "Date beyond knowledge cutoff: 2025-01-01 >= 2024-06-01"
```

### Example 3: Temporal Keywords

```
Input: "The latest climate report shows record temperatures."

┌─────────────────────────────────────┐
│ Step 1: Check URLs                  │
│ • URLs found: 0                     │
│ • Decision: Continue                │
└──────────┬──────────────────────────┘
           v
┌─────────────────────────────────────┐
│ Step 2: Extract Dates               │
│ • Dates found: None                 │
│ • Decision: Continue                │
└──────────┬──────────────────────────┘
           v
┌─────────────────────────────────────┐
│ Step 3: Check Keywords              │
│ • Keywords found: "latest"          │
│ • Decision: Use Web Research        │
└──────────┬──────────────────────────┘
           v
    ┌──────────────────────┐
    │ FactCheckerPipeline  │ ✅
    │ (Web Research)       │
    └──────────────────────┘

Result: "Temporal keywords suggest recent/current events"
```

### Example 4: URLs Provided

```
Input: "Company X reported record profits."
URLs: ["https://example.com/earnings"]

┌─────────────────────────────────────┐
│ Step 1: Check URLs                  │
│ • URLs found: 1                     │
│ • Decision: Use Web Research        │
└──────────┬──────────────────────────┘
           v
    ┌──────────────────────┐
    │ FactCheckerPipeline  │ ✅
    │ (Web Research)       │
    │ + Priority URLs      │
    └──────────────────────┘

Result: "URLs provided (1 URLs found)"
```

## Priority URL Processing Flow

```
FactCheckerPipeline receives priority_urls
        |
        v
┌──────────────────────────────┐
│ FireJudgeModule              │
│ (Iterative Research)         │
└──────┬───────────────────────┘
       |
       v
┌──────────────────────────────┐
│ ResearchAgentModule          │
│ Iteration 1 (with URLs)      │
└──────┬───────────────────────┘
       |
       v
┌──────────────────────────────────────────┐
│ Priority URL Processing                  │
│                                          │
│ for each priority_url (up to max_visits):│
│   1. Scrape URL                          │
│   2. Extract evidence                    │
│   3. Add to evidence list                │
│   4. Check if strong evidence found      │
│                                          │
│ Budget Remaining?                        │
│   YES → Continue to Web Search           │
│   NO  → Return evidence                  │
└──────┬───────────────────────────────────┘
       |
       v (if budget remains)
┌──────────────────────────────┐
│ Web Search                   │
│                              │
│ 1. Execute search query      │
│ 2. LLM selects pages         │
│ 3. Scrape selected pages     │
│ 4. Extract evidence          │
└──────┬───────────────────────┘
       |
       v
┌──────────────────────────────┐
│ Combine Evidence             │
│ • Priority URL evidence      │
│ • Web search evidence        │
└──────┬───────────────────────┘
       |
       v
   Return to FireJudge
```

## Date Extraction Process

```
Input Text
    |
    v
┌─────────────────────────────────┐
│ Apply Regex Patterns:           │
│                                 │
│ 1. YYYY-MM-DD                  │
│    "2025-01-15"                │
│                                 │
│ 2. Month DD, YYYY              │
│    "January 15, 2025"          │
│                                 │
│ 3. Mon DD, YYYY                │
│    "Jan 15, 2025"              │
│                                 │
│ 4. DD Month YYYY               │
│    "15 January 2025"           │
│                                 │
│ 5. Year only                   │
│    "in 2025", "of 2024"        │
└─────────┬───────────────────────┘
          |
          v
┌─────────────────────────────────┐
│ Parse Matched Strings           │
│ • Convert month names to nums   │
│ • Validate dates                │
│ • Handle edge cases             │
└─────────┬───────────────────────┘
          |
          v
    List[datetime]
```

## Temporal Keyword Detection

```
Input Text (lowercased)
    |
    v
┌─────────────────────────────────┐
│ Check for Keywords:             │
│                                 │
│ Time References:                │
│ • today, yesterday, tomorrow    │
│ • now, present, current         │
│                                 │
│ Relative Time:                  │
│ • this/last/next week/month/yr │
│                                 │
│ Recency:                        │
│ • recent, recently, latest      │
│ • upcoming                      │
│                                 │
│ Post-Cutoff Years:              │
│ • 2024, 2025, 2026             │
└─────────┬───────────────────────┘
          |
          v
     Boolean Result
   (True if any match)
```

## URL Extraction

```
Input Text
    |
    v
┌─────────────────────────────────┐
│ Apply Regex Pattern:            │
│                                 │
│ https?://[^\s<>"{}|\\^`\[\]]+  │
│                                 │
│ Matches:                        │
│ • http://example.com            │
│ • https://site.com/path?q=1    │
│ • https://news.org/article     │
└─────────┬───────────────────────┘
          |
          v
      List[str]
    (extracted URLs)
```

## Complete System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                              │
│  statement: str                                                 │
│  urls: Optional[list[str]]                                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            v
┌─────────────────────────────────────────────────────────────────┐
│                   TEMPORAL ROUTER MODULE                        │
│                                                                 │
│  1. _extract_urls(statement) → list[str]                       │
│  2. _extract_dates(statement) → list[datetime]                 │
│  3. _has_temporal_keywords(statement) → bool                   │
│  4. _should_use_web_research(...) → (bool, str)                │
│  5. Route to appropriate module                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              v                           v
┌──────────────────────┐      ┌──────────────────────────┐
│   JUDGE MODULE       │      │  FACTCHECKER PIPELINE    │
│   (Fast Path)        │      │  (Web Research Path)     │
│                      │      │                          │
│  1 API call          │      │  1. ClaimExtractor       │
│  ~1-3 seconds        │      │  2. For each claim:      │
│  Low cost            │      │     • FireJudge          │
│                      │      │       (iterative)        │
│  Returns:            │      │     • ResearchAgent      │
│  • verdict           │      │       - Priority URLs    │
│  • confidence        │      │       - Web Search       │
│  • reasoning         │      │     • Summarizer         │
│                      │      │  3. Aggregator           │
│                      │      │                          │
│                      │      │  15-30 API calls         │
│                      │      │  ~10-30 seconds          │
│                      │      │  Higher cost             │
│                      │      │                          │
│                      │      │  Returns:                │
│                      │      │  • claims                │
│                      │      │  • claim_results         │
│                      │      │  • verdict               │
│                      │      │  • confidence            │
│                      │      │  • reasoning             │
└──────────┬───────────┘      └────────┬─────────────────┘
           │                           │
           └──────────┬────────────────┘
                      │
                      v
┌─────────────────────────────────────────────────────────────────┐
│                      UNIFIED RESULT                             │
│                                                                 │
│  • statement: str                                              │
│  • overall_verdict: str                                        │
│  • confidence: float                                           │
│  • reasoning: str                                              │
│  • route_decision: "judge" | "pipeline"                        │
│  • route_reason: str                                           │
│  • claims: list[str] (pipeline only)                           │
│  • claim_results: list[...] (pipeline only)                    │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Comparison

```
Historical Statement (e.g., "Apollo 11 landed in 1969")
═══════════════════════════════════════════════════════

WITHOUT ROUTER:
    FactCheckerPipeline → ~15-30 API calls → ~$0.10 → ~20s
                         (unnecessary research)

WITH ROUTER:
    JudgeModule → 1 API call → ~$0.01 → ~2s
                  (90% cost reduction, 10x faster)

────────────────────────────────────────────────────────

Recent Statement (e.g., "In 2025, tech layoffs increased")
══════════════════════════════════════════════════════════

WITHOUT ROUTER:
    FactCheckerPipeline → ~15-30 API calls → ~$0.10 → ~20s
                         (necessary research)

WITH ROUTER:
    FactCheckerPipeline → ~15-30 API calls → ~$0.10 → ~20s
                          (same - research needed)

═══════════════════════════════════════════════════════════

Overall Improvement (50% historical, 50% recent):
    • Average cost reduction: 45%
    • Average latency reduction: 5x
    • Accuracy: Maintained or improved
```

## Monitoring Output Example

```
============================================================
TEMPORAL ROUTING DECISION
============================================================
Statement: In January 2025, tech companies announced...
URLs found: 0
Dates found: 1
  - 2025-01-01
Route: FactCheckerPipeline (with web research)
Reason: Date beyond knowledge cutoff: 2025-01-01 >= 2024-06-01
============================================================

[Web research begins...]
Processing 0 priority URLs before web search...
Search: tech companies layoffs January 2025
Visiting: https://techcrunch.com/layoffs-2025
[Evidence extracted...]

[Result]
Verdict: SUPPORTED
Confidence: 0.85
```
