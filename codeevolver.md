# Fact-Checking System Architecture

## High-Level Purpose
This is an automated fact-checking system built on the DSPy framework that evaluates the factual correctness of textual statements. The system offers two approaches: a **simple judge** that relies solely on LLM knowledge, and a **full pipeline** that performs iterative web research to verify claims with external evidence.

## Key Modules and Responsibilities

### Entry Point: `JudgeModule` (Simple Variant)
- **Location**: `src/factchecker/simple/modules/judge_module.py`
- **Purpose**: Barebones fact checker that directly evaluates statements without external research
- **Approach**: Uses DSPy's ChainOfThought reasoning to produce a verdict based on LLM's internal knowledge
- **Output**: Returns verdict (SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS), confidence score, and reasoning

### Full Pipeline: `FactCheckerPipeline`
- **Location**: `src/factchecker/modules/fact_checker_pipeline.py`
- **Components**:
  1. **ClaimExtractorModule**: Decomposes statements into individual verifiable claims
  2. **FireJudgeModule**: Iteratively evaluates each claim using the FIRE (Fact-checking with Iterative Research and Evaluation) approach
  3. **ResearchAgentModule**: Conducts web searches and scrapes pages using Serper and Firecrawl APIs
  4. **AggregatorModule**: Combines claim-level verdicts into an overall statement verdict using priority logic

## Data Flow

1. **Input**: Raw textual statement to be fact-checked
2. **Claim Extraction**: Statement is decomposed into atomic claims
3. **Iterative Research Loop** (per claim, max 3 iterations):
   - Judge evaluates current evidence
   - If insufficient, generates search query
   - Research agent fetches and summarizes relevant web pages
   - New evidence is appended for next iteration
4. **Aggregation**: Claim verdicts are combined with priority rules:
   - Any refuted claim → CONTAINS_REFUTED_CLAIMS
   - Any unsupported claim → CONTAINS_UNSUPPORTED_CLAIMS
   - All supported → SUPPORTED
5. **Output**: Final verdict with confidence, reasoning, and evidence trail

## Metric Being Optimized

**Primary Metric**: `gepa_metric` (defined in `src/optimizer/gepa_optimize.py`)
- **Target**: Maximize F1 score for detecting REFUTED claims
- **Scoring Logic**:
  - Correct predictions: score = 1.0
  - UNKNOWN predictions: score = 0.5 (neutral, acceptable when evidence is lacking)
  - Incorrect predictions: score = 0.0
- **Optimization Method**: GEPA (Generalized Evolutionary Prompt Adaptation) with reflective optimization
- **Evaluation Dataset**: FacTool QA benchmark (train/val/test split)
- **Secondary Metrics**: REFUTED precision/recall, SUPPORTED precision, overall accuracy

The system uses DSPy's GEPA optimizer to automatically improve prompts and reasoning chains through evolutionary search with LLM-based reflection on training examples.
