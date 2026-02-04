# FactChecker System Architecture

## Overview
The **JudgeModule** is a lightweight fact-checking system built with DSPy that evaluates statements for factual correctness. It represents a simplified alternative to the full FactCheckerPipeline, providing direct LLM-based judgment without external research or evidence gathering.

## Key Modules

### 1. **JudgeModule** (Entry Point)
- **Location**: `src/factchecker/simple/modules/judge_module.py`
- **Purpose**: Barebones fact checker that judges statements using only LLM knowledge
- **Components**: 
  - Uses DSPy's `ChainOfThought` reasoning with the `Judge` signature
  - No claim extraction, web search, or evidence gathering
- **Output**: Verdict (SUPPORTED/CONTAINS_UNSUPPORTED_CLAIMS/CONTAINS_REFUTED_CLAIMS), confidence score (0.0-1.0), and reasoning

### 2. **Judge Signature**
- **Location**: `src/factchecker/simple/signatures/judge.py`
- **Purpose**: Defines the LLM interaction schema for direct statement evaluation
- **Input**: Statement to evaluate
- **Outputs**: Reasoning, verdict (3 categories), confidence score

### 3. **Full FactCheckerPipeline** (Alternative Implementation)
- **Location**: `src/factchecker/modules/fact_checker_pipeline.py`
- **Purpose**: Complex multi-stage pipeline with:
  1. **ClaimExtractorModule**: Breaks statements into individual claims
  2. **ResearchAgentModule**: Performs web searches via Serper API and scrapes pages via Firecrawl
  3. **FireJudgeModule**: Iteratively evaluates claims with evidence (max 3 iterations, max 3 pages/query)
  4. **AggregatorModule**: Combines claim-level verdicts into overall statement verdict

## Data Flow

**Simple Path (JudgeModule)**:
```
Statement → Judge (ChainOfThought) → Verdict + Confidence + Reasoning
```

**Full Pipeline Path**:
```
Statement → Claim Extraction → Research Agent (Web Search/Scraping) 
→ Fire Judge (Iterative Evaluation) → Aggregation → Final Verdict
```

## Metric Being Optimized

**Primary Metric**: `gepa_metric` in `src/codeevolver/metric.py` (re-exports from `src/optimizer/gepa_optimize.py`)

**Optimization Goal**:
- **Maximize F1 score** for the **REFUTED** class (primary focus)
- **Secondary consideration**: Precision of **SUPPORTED** class
- Uses GEPA (Generalized Evolutionary Prompt Adaptation) optimizer from DSPy

**Scoring Logic**:
- Correct prediction (matches gold label): **1.0**
- UNKNOWN prediction (neutral): **0.5** (acceptable when evidence is lacking)
- Incorrect prediction: **0.0**

**Key Performance Indicators**:
- REFUTED F1 score (balance of precision and recall for detecting false statements)
- REFUTED precision/recall
- SUPPORTED precision (avoid false positives)
- Overall accuracy

The system is designed to prioritize catching false information (high REFUTED recall) while maintaining confidence in supported claims (high SUPPORTED precision).
