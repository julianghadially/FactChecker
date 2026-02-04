# SmartJudgeModule Routing Diagram

## High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     SmartJudgeModule.forward()                  │
│                  (statement, urls=None)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  URLs provided? │
                    └────────┬────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
                YES                     NO
                 │                       │
                 ▼                       ▼
    ┌────────────────────────┐  ┌──────────────────────┐
    │  1. Scrape URLs        │  │ Temporal claim       │
    │  2. Format as evidence │  │ detected?            │
    │  3. Call Pipeline with │  └──────────┬───────────┘
    │     initial_evidence   │             │
    └────────────┬───────────┘  ┌──────────┴──────────┐
                 │              │                     │
                 │             YES                   NO
                 │              │                     │
                 │              ▼                     ▼
                 │    ┌───────────────────┐  ┌──────────────────┐
                 │    │ Call Pipeline     │  │ Try JudgeModule  │
                 │    │ for web research  │  │ first            │
                 │    └─────────┬─────────┘  └────────┬─────────┘
                 │              │                     │
                 │              │                     ▼
                 │              │          ┌────────────────────────┐
                 │              │          │ Confidence < threshold │
                 │              │          │ OR verdict =           │
                 │              │          │ UNSUPPORTED?           │
                 │              │          └──────────┬─────────────┘
                 │              │                     │
                 │              │          ┌──────────┴──────────┐
                 │              │          │                     │
                 │              │         YES                   NO
                 │              │          │                     │
                 │              │          ▼                     ▼
                 │              │  ┌──────────────────┐  ┌─────────────┐
                 │              │  │ Fallback to      │  │ Return      │
                 │              │  │ Pipeline for     │  │ JudgeModule │
                 │              │  │ web research     │  │ result      │
                 │              │  └────────┬─────────┘  └──────┬──────┘
                 │              │           │                   │
                 ▼              ▼           ▼                   ▼
    ┌────────────────────────────────────────────────────────────┐
    │              Return dspy.Prediction                        │
    │  (statement, verdict, confidence, reasoning,               │
    │   routing_decision, [claims, claim_results])               │
    └────────────────────────────────────────────────────────────┘
```

## Detailed Decision Tree

```
SmartJudgeModule.forward(statement, urls)
│
├─[Route 1: URL-Based]──────────────────────────────────────┐
│  Condition: urls is not None and len(urls) > 0            │
│  Action:                                                   │
│    1. For each URL in urls:                               │
│       - Scrape with FirecrawlService                      │
│       - Format as evidence section                        │
│    2. Combine all evidence sections                       │
│    3. Call FactCheckerPipeline(                           │
│         statement=statement,                              │
│         initial_evidence=combined_evidence                │
│       )                                                   │
│  Time: ~30-60 seconds                                     │
│  Cost: ~$0.05-0.15                                        │
│  Return: Pipeline result + routing_decision               │
└───────────────────────────────────────────────────────────┘
│
├─[Route 2: Temporal Detection]─────────────────────────────┐
│  Condition: URLs not provided AND                         │
│             temporal_detector finds recent/future dates   │
│  Action:                                                  │
│    1. Call TemporalDetector (lightweight LLM call)       │
│    2. If requires_recent_knowledge == True:              │
│       - Call FactCheckerPipeline(statement)              │
│  Time: ~31-61 seconds (temporal check + pipeline)        │
│  Cost: ~$0.055-0.155                                      │
│  Return: Pipeline result + routing_decision               │
└───────────────────────────────────────────────────────────┘
│
└─[Route 3: Confidence-Based Fallback]──────────────────────┐
   Condition: URLs not provided AND no temporal claims      │
   Action:                                                  │
     1. Call JudgeModule(statement)                         │
     2. Check result:                                       │
        ├─ If confidence >= threshold (default 0.6) AND    │
        │  verdict != CONTAINS_UNSUPPORTED_CLAIMS:          │
        │  └─> Return JudgeModule result                   │
        │      Time: ~1-3 seconds                           │
        │      Cost: ~$0.001                                │
        │                                                   │
        └─ If confidence < threshold OR                    │
           verdict == CONTAINS_UNSUPPORTED_CLAIMS:          │
           └─> Fallback to FactCheckerPipeline              │
               Time: ~31-63 seconds (judge + pipeline)      │
               Cost: ~$0.051-0.151                          │
   Return: Best available result + routing_decision         │
────────────────────────────────────────────────────────────┘
```

## Temporal Detection Logic

```
TemporalDetector(statement)
│
├─ Checks for:
│  ├─ Year references >= 2024
│  │  Examples: "In 2024", "2025 projections", "by 2026"
│  │
│  ├─ Future date indicators
│  │  Examples: "next year", "upcoming", "will be"
│  │
│  ├─ Recent temporal phrases
│  │  Examples: "recently", "this year", "currently"
│  │
│  └─ Status claims (time-sensitive)
│     Examples: "current president", "latest version"
│
└─ Returns:
   ├─ reasoning: Explanation of detection
   └─ requires_recent_knowledge: boolean
      ├─ True  → Routes to FactCheckerPipeline
      └─ False → Continues to confidence check
```

## Confidence-Based Fallback Logic

```
JudgeModule Result Analysis
│
├─ Extract:
│  ├─ confidence: float (0.0 - 1.0)
│  └─ verdict: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
│
└─ Decision:
   │
   ├─[High Confidence Path]──────────────────────────────┐
   │  Condition: confidence >= threshold (default 0.6)   │
   │             AND verdict != CONTAINS_UNSUPPORTED     │
   │  Action: Return JudgeModule result immediately      │
   │  Rationale: LLM is confident, no need for research  │
   └─────────────────────────────────────────────────────┘
   │
   └─[Fallback Path]──────────────────────────────────────┐
      Condition: confidence < threshold OR                │
                 verdict == CONTAINS_UNSUPPORTED_CLAIMS   │
      Action: Call FactCheckerPipeline for verification   │
      Rationale: LLM uncertain or lacks knowledge         │
      └─────────────────────────────────────────────────────┘
```

## URL Pre-Seeding Process

```
_scrape_urls_as_evidence(urls)
│
└─ For each URL:
   │
   ├─ Call FirecrawlService.scrape(url)
   │  ├─ Success:
   │  │  └─ Format:
   │  │     "--- Pre-seeded Evidence from {url} ---"
   │  │     "Title: {title}"
   │  │     "Content: {markdown}"
   │  │
   │  └─ Failure:
   │     └─ Format:
   │        "--- Failed to scrape {url} ---"
   │        "Error: {error}"
   │
   └─ Combine all sections with "\n\n"
      │
      └─ Pass to FactCheckerPipeline(
            statement=statement,
            initial_evidence=formatted_evidence
         )
         │
         └─ Pipeline propagates to FireJudgeModule
            │
            └─ FireJudgeModule uses evidence in first iteration
               (before any web searches)
```

## Return Value Structure

```
dspy.Prediction
├─ statement: str (input statement)
├─ overall_verdict: str
│  ├─ "SUPPORTED"
│  ├─ "CONTAINS_UNSUPPORTED_CLAIMS"
│  └─ "CONTAINS_REFUTED_CLAIMS"
├─ confidence: float (0.0 - 1.0)
├─ reasoning: str (explanation)
├─ routing_decision: str (describes path taken)
│  Examples:
│  ├─ "URLs provided (2 URLs) - routing to FactCheckerPipeline with pre-seeded evidence"
│  ├─ "Temporal claim detected (recent/future dates) - routing to FactCheckerPipeline for web research"
│  ├─ "No URLs or temporal claims - trying JudgeModule first -> High confidence (0.92) - using JudgeModule result"
│  └─ "No URLs or temporal claims - trying JudgeModule first -> Falling back to FactCheckerPipeline (low confidence (0.45 < 0.6))"
│
└─ Optional (only if pipeline was used):
   ├─ claims: list[str] (extracted claims)
   └─ claim_results: list[JudgmentResult] (per-claim verdicts)
```

## Performance Characteristics

```
Route Performance Comparison
│
├─[JudgeModule Only (Fast Path)]
│  Latency:  1-3 seconds
│  Cost:     ~$0.001
│  Quality:  Good for known facts
│  LLM Calls: 1 (ChainOfThought)
│
├─[FactCheckerPipeline (Full Path)]
│  Latency:  30-60 seconds
│  Cost:     ~$0.05-0.15
│  Quality:  Best (web-verified)
│  LLM Calls: ~5-10 (claim extraction, page selection,
│              evidence summarization, iterative judgment,
│              aggregation)
│
└─[Hybrid (Judge + Fallback)]
   Latency:  2-63 seconds (depends on fallback)
   Cost:     $0.001 (no fallback) to $0.151 (with fallback)
   Quality:  Best of both worlds
   LLM Calls: 1-11 (judge + optional pipeline)
```

## Configuration Impact

```
confidence_threshold Parameter
│
├─ Lower (e.g., 0.4)
│  ├─ More JudgeModule usage
│  ├─ Faster overall
│  ├─ Cheaper overall
│  └─ May miss complex cases
│
├─ Default (0.6)
│  ├─ Balanced approach
│  ├─ Good mix of speed and accuracy
│  └─ Recommended starting point
│
└─ Higher (e.g., 0.8)
   ├─ More Pipeline usage
   ├─ Slower but more thorough
   ├─ More expensive
   └─ Better for critical applications
```

## Example Routing Scenarios

```
Scenario Examples
│
├─ "Water boils at 100°C at sea level"
│  Path: Route 3 (Judge only)
│  Reason: High confidence historical fact
│  Time: ~2 seconds
│
├─ "In 2024, global temperatures rose 1.5°C"
│  Path: Route 2 (Temporal)
│  Reason: Contains 2024 reference
│  Time: ~35 seconds
│
├─ "Python is popular" + URLs=[python.org]
│  Path: Route 1 (URL pre-seed)
│  Reason: URLs provided
│  Time: ~40 seconds
│
└─ "The Eiffel Tower has 18,038 iron pieces"
   Path: Route 3 (Judge → Fallback)
   Reason: Low confidence on specific number
   Time: ~45 seconds (2s judge + 43s fallback)
```
