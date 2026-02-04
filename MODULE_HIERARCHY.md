# Fact-Checker Module Hierarchy

## Overview

The fact-checker system consists of three tiers of modules, each building on the previous tier to provide different levels of fact-checking capability.

```
┌─────────────────────────────────────────────────────────────┐
│                    TIER 3: ADAPTIVE                         │
│                  (Intelligent Routing)                      │
│                                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │         AdaptiveJudgeModule (NEW!)               │      │
│  │  Intelligent routing based on confidence         │      │
│  │  Auto-fallback to pipeline when needed           │      │
│  └──────────────┬────────────────────────┬──────────┘      │
└─────────────────┼────────────────────────┼─────────────────┘
                  │                        │
       ┌──────────▼──────────┐  ┌─────────▼────────────┐
       │                     │  │                       │
┌──────┴─────────────────────┴──┴───────────────────────┴─────┐
│              TIER 2: RESEARCH-ENABLED                        │
│             (Full Fact-Checking)                             │
│                                                              │
│  ┌────────────────────────────────────────────────┐         │
│  │         FactCheckerPipeline                    │         │
│  │  Complete fact-checking with web research      │         │
│  │  1. Extract claims                             │         │
│  │  2. Research each claim                        │         │
│  │  3. Aggregate results                          │         │
│  └────┬────────────────┬───────────────┬──────────┘         │
└───────┼────────────────┼───────────────┼────────────────────┘
        │                │               │
        ▼                ▼               ▼
  ┌──────────┐   ┌──────────────┐   ┌─────────┐
  │ Claim    │   │ FireJudge    │   │ Aggre-  │
  │ Extractor│   │ Module       │   │ gator   │
  └────┬─────┘   └──────┬───────┘   └─────────┘
       │                │
       │         ┌──────▼──────┐
       │         │ Research    │
       │         │ Agent       │
       │         └─────────────┘
       │
┌──────┴──────────────────────────────────────────────────────┐
│              TIER 1: SIMPLE                                  │
│             (Fast, No Research)                              │
│                                                              │
│  ┌────────────────────────────────────────────────┐         │
│  │              JudgeModule                       │         │
│  │  Fast fact-checking using LLM knowledge only   │         │
│  │  No claims extraction, no web search           │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Module Comparison

| Module | Speed | Accuracy | Research | Use Case |
|--------|-------|----------|----------|----------|
| **JudgeModule** | ⚡⚡⚡ | ✅ Good | ❌ No | Quick checks, known facts |
| **FactCheckerPipeline** | 🐢 Slow | ✅✅✅ Excellent | ✅ Yes | Critical verification |
| **AdaptiveJudgeModule** | ⚡/🐢 | ✅✅ Very Good | 🔄 Auto | General purpose |

## Detailed Module Descriptions

### Tier 1: Simple Modules

#### JudgeModule
```python
from src.factchecker.simple.modules.judge_module import JudgeModule

judge = JudgeModule()
result = judge(statement="Water boils at 100°C")
# Fast (1-2s), no web research
```

**Features:**
- Direct LLM evaluation
- Single API call
- Returns: verdict, confidence, reasoning
- Best for: Known facts, quick checks

**Output:**
```python
{
    "statement": "Water boils at 100°C",
    "overall_verdict": "SUPPORTED",
    "confidence": 0.95,
    "reasoning": "This is a well-known fact..."
}
```

---

### Tier 2: Research-Enabled Modules

#### FactCheckerPipeline
```python
from src.factchecker.modules import FactCheckerPipeline

pipeline = FactCheckerPipeline(
    max_judge_iterations=3,
    max_page_visits=3
)
result = pipeline(statement="Company X Q4 2024 revenue was $523M")
# Slow (10-30s), performs web research
```

**Features:**
- Extracts individual claims
- Iterative web research per claim
- Evidence gathering and evaluation
- Aggregates claim verdicts
- Best for: Critical verification, complex statements

**Internal Flow:**
1. **ClaimExtractorModule**: Breaks statement into atomic claims
2. **FireJudgeModule**: Evaluates each claim with iterative research
   - Uses **ResearchAgentModule** for web search and scraping
3. **AggregatorModule**: Combines claim verdicts into overall verdict

**Output:**
```python
{
    "statement": "Company X Q4 2024 revenue was $523M",
    "claims": ["Company X Q4 2024 revenue was $523M"],
    "claim_results": [
        {
            "claim": "...",
            "verdict": "supported",
            "evidence_summary": "...",
            "search_queries": [...],
            "iterations": 2
        }
    ],
    "overall_verdict": "SUPPORTED",
    "confidence": 0.92,
    "reasoning": "Based on verified financial reports..."
}
```

#### Supporting Modules

**ClaimExtractorModule**
- Breaks complex statements into atomic claims
- Each claim can be independently verified

**FireJudgeModule**
- Iterative claim evaluation
- Generates search queries
- Evaluates evidence
- Stops when confident or max iterations reached

**ResearchAgentModule**
- Performs web searches (via Serper)
- Scrapes and extracts page content (via Firecrawl)
- Returns relevant evidence

**AggregatorModule**
- Combines claim-level verdicts
- Determines overall statement verdict
- Provides confidence and reasoning

---

### Tier 3: Adaptive Modules

#### AdaptiveJudgeModule (NEW!)
```python
from src.factchecker.modules import AdaptiveJudgeModule

adaptive = AdaptiveJudgeModule(
    confidence_threshold=0.7,
    enable_fallback=True
)
result = adaptive(statement="Some claim")
# Automatically chooses fast or slow path
```

**Features:**
- Intelligent routing between JudgeModule and FactCheckerPipeline
- Confidence-based decision making
- Lazy pipeline initialization
- Transparent fallback indication
- Best for: General purpose, balanced speed/accuracy

**Decision Logic:**
```
IF (verdict == "CONTAINS_UNSUPPORTED_CLAIMS"
    AND confidence < threshold
    AND fallback_enabled):
    → Use FactCheckerPipeline (slow, thorough)
ELSE:
    → Use JudgeModule (fast, sufficient)
```

**Output (No Fallback):**
```python
{
    "statement": "Water boils at 100°C",
    "overall_verdict": "SUPPORTED",
    "confidence": 0.95,
    "reasoning": "...",
    "fallback_triggered": False
}
```

**Output (With Fallback):**
```python
{
    "statement": "Company X Q4 2024 revenue was $523M",
    "overall_verdict": "SUPPORTED",
    "confidence": 0.92,
    "reasoning": "...",
    "fallback_triggered": True,
    "claims": [...],
    "claim_results": [...]
}
```

## Usage Recommendations

### Choose JudgeModule When:
- ✅ Checking well-known facts
- ✅ Speed is critical
- ✅ Cost is a major concern
- ✅ Don't need external verification
- ✅ Processing high volume of statements

**Example:** Content moderation, quick fact-checking for obvious claims

### Choose FactCheckerPipeline When:
- ✅ Accuracy is paramount
- ✅ Need evidence and sources
- ✅ Verifying recent/obscure claims
- ✅ Complex statements with multiple claims
- ✅ Critical applications (medical, legal, financial)

**Example:** Investigative journalism, academic fact-checking, legal research

### Choose AdaptiveJudgeModule When:
- ✅ Want balanced speed and accuracy
- ✅ Mix of known and unknown facts
- ✅ Uncertain about complexity
- ✅ Want automatic optimization
- ✅ General-purpose fact-checking

**Example:** News article verification, chatbot fact-checking, user-generated content

## Configuration Patterns

### Pattern 1: Speed-Optimized
```python
# Use simplest module
judge = JudgeModule()

# Or adaptive with fallback disabled
adaptive = AdaptiveJudgeModule(enable_fallback=False)
```

### Pattern 2: Accuracy-Optimized
```python
# Use full pipeline with thorough research
pipeline = FactCheckerPipeline(
    max_judge_iterations=5,
    max_page_visits=5
)

# Or adaptive with aggressive fallback
adaptive = AdaptiveJudgeModule(
    confidence_threshold=0.9,
    max_judge_iterations=5,
    max_page_visits=5
)
```

### Pattern 3: Balanced (Recommended)
```python
# Use adaptive with defaults
adaptive = AdaptiveJudgeModule()
```

### Pattern 4: Cost-Conscious
```python
# Adaptive with conservative fallback
adaptive = AdaptiveJudgeModule(
    confidence_threshold=0.6,  # Less likely to fallback
    max_judge_iterations=2,    # Limit research depth
    max_page_visits=2
)
```

## API Requirements

### Always Required
- **LLM API**: OpenAI, Anthropic, or other DSPy-supported provider
  - `OPENAI_API_KEY` (or equivalent)

### Required for Research (Pipeline and Fallback)
- **Web Search**: Serper API
  - `SERPER_API_KEY`
- **Web Scraping**: Firecrawl API
  - `FIRECRAWL_API_KEY`

### API Usage by Module

| Module | LLM Calls | Serper Calls | Firecrawl Calls |
|--------|-----------|--------------|-----------------|
| JudgeModule | 1 | 0 | 0 |
| FactCheckerPipeline | 5-15 | 1-5 per claim | 1-5 per claim |
| AdaptiveJudgeModule | 1 or 5-15 | 0 or 1-5 | 0 or 1-5 |

## Performance Characteristics

### JudgeModule
- **Latency**: 1-2 seconds
- **Cost**: $0.001 - $0.01 per request
- **Accuracy**: 70-85% (depends on LLM knowledge)

### FactCheckerPipeline
- **Latency**: 10-30 seconds
- **Cost**: $0.05 - $0.20 per request
- **Accuracy**: 85-95% (with web verification)

### AdaptiveJudgeModule
- **Latency**: 1-2s (fast path) or 10-30s (slow path)
- **Cost**: Varies based on fallback rate
- **Accuracy**: 75-90% (adaptive)

**Fallback Rate Impact:**
- 10% fallback: ~2s average, ~$0.01 per request
- 50% fallback: ~15s average, ~$0.05 per request
- 90% fallback: ~25s average, ~$0.15 per request

## Example Workflow

### Simple Flow (JudgeModule)
```python
judge = JudgeModule()
result = judge(statement="The sky is blue")
print(result.overall_verdict)
```

### Research Flow (FactCheckerPipeline)
```python
pipeline = FactCheckerPipeline()
result = pipeline(statement="Company X revenue exceeded $500M in Q4 2024")

for claim_result in result.claim_results:
    print(f"Claim: {claim_result.claim}")
    print(f"Evidence: {claim_result.evidence_summary}")
```

### Adaptive Flow (AdaptiveJudgeModule)
```python
adaptive = AdaptiveJudgeModule()
result = adaptive(statement="Some claim")

if result.fallback_triggered:
    print("Performed web research")
    print(f"Analyzed {len(result.claims)} claims")
else:
    print("Used LLM knowledge only")

print(f"Verdict: {result.overall_verdict}")
```

## Module Dependencies

```
AdaptiveJudgeModule
├── JudgeModule (simple)
└── FactCheckerPipeline
    ├── ClaimExtractorModule
    ├── FireJudgeModule
    │   └── ResearchAgentModule
    │       ├── SerperService
    │       └── FirecrawlService
    └── AggregatorModule
```

## File Locations

```
src/factchecker/
├── simple/
│   └── modules/
│       └── judge_module.py           # Tier 1: JudgeModule
├── modules/
│   ├── fact_checker_pipeline.py      # Tier 2: FactCheckerPipeline
│   ├── claim_extractor_module.py     # Supporting
│   ├── fire_judge_module.py          # Supporting
│   ├── research_agent_module.py      # Supporting
│   ├── aggregator_module.py          # Supporting
│   └── adaptive_judge_module.py      # Tier 3: AdaptiveJudgeModule (NEW!)
└── services/
    ├── serper_service.py              # Web search
    └── firecrawl_service.py           # Web scraping
```

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install dspy-ai openai firecrawl-py
   ```

2. **Set API keys:**
   ```bash
   export OPENAI_API_KEY="..."
   export SERPER_API_KEY="..."       # Optional
   export FIRECRAWL_API_KEY="..."    # Optional
   ```

3. **Choose your module:**
   ```python
   import dspy
   from src.factchecker.modules import AdaptiveJudgeModule

   lm = dspy.LM('openai/gpt-4o-mini')
   dspy.configure(lm=lm)

   adaptive = AdaptiveJudgeModule()
   result = adaptive(statement="Your statement here")
   ```

4. **Check results:**
   ```python
   print(f"Verdict: {result.overall_verdict}")
   print(f"Confidence: {result.confidence}")
   print(f"Research performed: {result.fallback_triggered}")
   ```

## Documentation

- **AdaptiveJudgeModule**: `src/factchecker/modules/README_ADAPTIVE_JUDGE.md`
- **Quick Start**: `QUICK_START.md`
- **Implementation Summary**: `ADAPTIVE_JUDGE_SUMMARY.md`
- **Flowchart**: `src/factchecker/modules/ADAPTIVE_JUDGE_FLOWCHART.txt`

## Summary

The three-tier architecture provides flexibility for different use cases:

- **Tier 1 (Simple)**: Fast, cheap, no research
- **Tier 2 (Research)**: Thorough, accurate, with evidence
- **Tier 3 (Adaptive)**: Intelligent, balanced, automatic

Choose based on your needs for speed, accuracy, and cost.
