# FactChecker Architecture Summary

## High-Level Purpose
FactChecker is a DSPy-based fact verification system designed to assess the factual correctness of language model outputs. Unlike simple LLM-as-judge approaches, FactChecker grounds its judgments in external evidence through iterative web search and research. The entry point `JudgeModule` represents a simplified version that evaluates statements directly without external research, serving as a faster baseline.

## Key Modules and Responsibilities

### Core Pipeline (Full System)
1. **FactCheckerPipeline** - Orchestrates the complete fact-checking flow
2. **ClaimExtractorModule** - Breaks down input statements into individual verifiable claims
3. **FireJudgeModule** - Iteratively evaluates each claim, deciding whether to request more evidence or return a verdict
4. **ResearchAgentModule** - Orchestrates web search and evidence gathering workflow
5. **AggregatorModule** - Combines individual claim verdicts into an overall statement verdict using priority logic (any refuted → CONTAINS_REFUTED; any unsupported → CONTAINS_UNSUPPORTED; all supported → SUPPORTED)

### Research Components
- **SerperService** - Google search integration via Serper API
- **PageSelector** - LLM-guided selection of which search result pages to visit (max 3 per query)
- **FirecrawlService** - Web scraping for page content
- **EvidenceSummarizer** - Extracts relevant facts from scraped pages

### Simple Alternative
- **JudgeModule** (entry point) - Barebones fact checker that judges statements directly using LLM knowledge without external research, web search, or evidence gathering

## Data Flow

1. **Input**: Statement (string)
2. **Claim Extraction**: Statement → Claims[] (multiple verifiable sub-claims)
3. **Iterative Evaluation** (per claim):
   - FireJudge analyzes claim with current evidence
   - If verdict reached → return judgment
   - If more info needed → Research cycle:
     - Serper search → results
     - PageSelector chooses relevant pages
     - Firecrawl scrapes content
     - EvidenceSummarizer extracts facts
     - Evidence fed back to FireJudge (repeat up to max_judge_iterations=3)
4. **Aggregation**: Individual claim verdicts → overall statement verdict
5. **Output**: FactCheckResult with verdict, confidence, reasoning, and claim-level details

## Metric Being Optimized

The system optimizes using **`gepa_metric`** (defined in `src.optimizer.gepa_optimize`), which implements a GEPA (Generative Evolutionary Prompt Augmentation) optimization strategy:

- **Primary Goal**: Maximize F1 score for the REFUTED class
- **Secondary Goal**: Maximize precision for the SUPPORTED class
- **Scoring Logic**:
  - Correct prediction: score = 1.0
  - UNKNOWN prediction (uncertain): score = 0.5 (neutral feedback, acceptable when evidence is insufficient)
  - Incorrect prediction: score = 0.0
- **Optimization Results**: GEPA optimization improved FactChecker from 91% to 96% accuracy on current event claims, with +10-18 percentage point gains in class-specific recall

The metric balances accuracy with the system's ability to confidently identify false information while avoiding overconfidence on uncertain claims.
