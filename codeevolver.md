# FactChecker Architecture Summary

## High-Level Purpose

FactChecker is a **DSPy-based fact verification system** that evaluates the factual correctness of statements by grounding judgments in external evidence through iterative web research. The system being optimized is `JudgeModule`, a simplified fact checker that makes verdicts directly using LLM knowledge without external research—serving as a faster alternative to the full research-based pipeline.

## Key Modules and Responsibilities

### Entry Point: `JudgeModule` (Simple Fact Checker)
- **Location**: `src/factchecker/simple/modules/judge_module.py`
- **Purpose**: Barebones fact checker that evaluates statements using only LLM knowledge
- **Components**:
  - Uses DSPy's `ChainOfThought` with the `Judge` signature
  - Takes a statement as input
  - Returns verdict (SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS), confidence score, and reasoning
  - No claim extraction, no web search, no evidence gathering

### Full Pipeline: `FactCheckerPipeline` (Research-Based)
For comparison, the full system includes:
- **ClaimExtractorModule**: Breaks statements into individual factual claims
- **FireJudgeModule**: Iteratively evaluates claims, requesting searches when needed
- **ResearchAgentModule**: Orchestrates web search (Serper), page selection, scraping (Firecrawl), and evidence extraction
- **AggregatorModule**: Combines claim-level verdicts using priority logic (any refuted → CONTAINS_REFUTED; any unsupported → CONTAINS_UNSUPPORTED; all supported → SUPPORTED)

## Data Flow

### Simple JudgeModule (Entry Point):
```
Statement → ChainOfThought(Judge Signature) → {verdict, confidence, reasoning}
```

### Full Pipeline (Context):
```
Statement → ClaimExtractor → Claims[] → FireJudge (iterative) ↔ ResearchAgent (search/scrape/evidence) → Aggregator → Final Verdict
```

## Metric Being Optimized

**Metric**: `gepa_metric` from `src/codeevolver/metric.py` (wraps `src/optimizer/gepa_optimize.py`)

**Optimization Strategy**:
- Primary: Maximize F1 score for REFUTED class detection
- Secondary: Maximize precision for SUPPORTED class predictions
- Scoring: Correct prediction = 1.0, UNKNOWN = 0.5 (neutral), Incorrect = 0.0
- Optimizer: GEPA (reflective prompt optimization) using DSPy
- Training: Uses FacTool-QA dataset with train/val/test splits

**Performance Impact**: GEPA optimization improved accuracy from 91% → 96% on current events, with 10-18 percentage point gains in class-specific recall (SUPPORTED: 49%→59%, REFUTED: 65%→83%).
