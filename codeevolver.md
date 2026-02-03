# Architecture Summary: JudgeModule Fact-Checking System

## High-Level Purpose
This system implements an AI-powered fact-checking pipeline built on the DSPy framework. The **JudgeModule** serves as a simplified entry point for rapid fact verification without external research, while the full **FactCheckerPipeline** provides comprehensive claim-by-claim verification using iterative web research. The system is designed to classify statements as SUPPORTED, CONTAINS_UNSUPPORTED_CLAIMS, or CONTAINS_REFUTED_CLAIMS.

## Key Modules and Responsibilities

### 1. **JudgeModule** (Entry Point - Simple Path)
- **Location**: `src/factchecker/simple/modules/judge_module.py`
- **Purpose**: Lightweight fact-checker that evaluates statements directly using LLM knowledge without external research
- **Components**: Uses `Judge` signature with `ChainOfThought` reasoning
- **Output**: Statement verdict, confidence score (0.0-1.0), and reasoning

### 2. **FactCheckerPipeline** (Full Pipeline)
- **Location**: `src/factchecker/modules/fact_checker_pipeline.py`
- **Purpose**: Comprehensive orchestrator implementing the complete fact-checking workflow
- **Sub-modules**:
  - **ClaimExtractorModule**: Decomposes statements into independently verifiable atomic claims
  - **FireJudgeModule**: Iteratively evaluates each claim using FIRE (Fact-checking with Iterative Research and Evaluation) approach with up to 3 search iterations
  - **ResearchAgentModule**: Conducts web searches and scrapes pages (via Serper/Firecrawl APIs)
  - **AggregatorModule**: Combines claim-level verdicts using priority logic (refuted > unsupported > supported)

### 3. **GEPA Optimizer**
- **Location**: `src/optimizer/gepa_optimize.py`
- **Purpose**: Optimizes prompts and pipeline behavior using DSPy's GEPA (Generalized Evolutionary Prompt Algorithm)
- **Integration**: Uses reflective optimization with configurable intensity (light/medium/heavy)

## Data Flow

```
Input Statement
    ↓
[Simple Path]                [Full Pipeline Path]
JudgeModule                  ClaimExtractorModule
    ↓                              ↓
Direct LLM                   Individual Claims
Judgment                          ↓
    ↓                        FireJudgeModule (per claim)
                                  ↓
                             Iterative Research Loop:
                             - Judge evaluates claim
                             - Requests search query
                             - ResearchAgent fetches evidence
                             - Repeat up to 3 times
                                  ↓
                             Claim Verdicts
                                  ↓
                             AggregatorModule
                                  ↓
Final Verdict (SUPPORTED / CONTAINS_UNSUPPORTED_CLAIMS / CONTAINS_REFUTED_CLAIMS)
+ Confidence Score + Reasoning
```

## Metric Being Optimized

**Primary Metric**: `gepa_metric` function in `src/codeevolver/metric/metric.py` (re-exports from `gepa_optimize.py`)

**Optimization Target**:
- **F1 Score** for the REFUTED class (primary focus)
- **Precision** for the SUPPORTED class (secondary)
- **Scoring Logic**:
  - Correct prediction: 1.0
  - UNKNOWN prediction (acceptable uncertainty): 0.5
  - Incorrect prediction: 0.0

The metric provides structured feedback for GEPA's reflective optimization process, enabling the system to learn better prompting strategies for maximizing fact-checking accuracy on the FacTool_QA benchmark dataset.
