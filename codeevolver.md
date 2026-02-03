# FactChecker Architecture Summary

## High-Level Purpose
FactChecker is a DSPy-based fact verification system that evaluates the factual correctness of statements by grounding judgments in external evidence through iterative web search. Unlike simple LLM-as-judge approaches, it performs systematic web research to verify or refute claims, reducing model bias and improving reliability on current events and verifiable facts.

## Entry Point: JudgeModule (Simple)
The entry point `src.factchecker.simple.modules.judge_module.JudgeModule` is a **barebones fact checker** that provides fast, direct verdict generation without external research. It uses DSPy's ChainOfThought with a Judge signature to evaluate statements purely based on LLM knowledge, outputting:
- `overall_verdict`: SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
- `confidence`: Float between 0.0-1.0
- `reasoning`: Explanation text

This serves as a simpler alternative to the full FactCheckerPipeline for cases where external research is not needed.

## Key Modules & Responsibilities

### 1. **FactCheckerPipeline** (Full System)
Complete orchestration flow combining all components:
- **ClaimExtractorModule**: Decomposes statements into individual factual claims
- **FireJudgeModule**: Iteratively evaluates each claim (max 3 iterations), deciding whether to return verdict or request more evidence
- **ResearchAgentModule**: Orchestrates web search pipeline when FireJudge needs evidence:
  - SerperService: Google search via Serper API
  - PageSelector (LLM-guided): Selects top 3 relevant pages
  - FirecrawlService: Scrapes page content
  - EvidenceSummarizer: Extracts relevant facts from scraped pages
- **AggregatorModule**: Combines claim-level verdicts using priority logic:
  1. Any refuted → CONTAINS_REFUTED
  2. Any unsupported → CONTAINS_UNSUPPORTED  
  3. All supported → SUPPORTED

### 2. **Optimization System (GEPA)**
- `gepa_optimize.py`: Runs DSPy GEPA (reflective optimization) to improve prompts/reasoning
- Uses train/val/test splits from FacTool-QA dataset
- Optimizes via reflection with configurable intensity (light/medium/heavy)
- Tracks with MLflow for experiment management

## Data Flow
```
Statement 
  → ClaimExtractor → Claims[]
  → For each claim:
      FireJudge (iterative):
        → Needs evidence? → ResearchAgent (search→select→scrape→summarize) → Evidence
        → Has verdict? → Verdict
  → Aggregator (priority logic) 
  → Final Verdict
```

## Metric Being Optimized
**Primary Metric**: `gepa_metric` in `src.codeevolver.metric.metric`

The metric optimizes **instance-level correctness** with three outcome scores:
- **Score 1.0** (Correct): `pred_label == gold.label`
- **Score 0.5** (Neutral): Predicting "UNKNOWN" when evidence is insufficient
- **Score 0.0** (Incorrect): Wrong prediction

**Evaluation Focus**: 
- **REFUTED F1 Score**: Primary optimization target (detect false claims)
- **SUPPORTED Precision**: Secondary metric (avoid false positives)
- **REFUTED Recall**: Maximize detection of false information

The GEPA optimizer uses reflective feedback loops to improve the system's ability to correctly classify claims while avoiding over-prediction, with special handling for "UNKNOWN" predictions to balance precision/recall trade-offs.

**Performance**: After GEPA optimization, the system achieves 96% accuracy on news claims with 83% REFUTED recall and 59% SUPPORTED recall, significantly outperforming the baseline GPT-5-mini (96% accuracy but only 23% REFUTED recall).
