# Architecture Summary: JudgeModule Fact-Checking System

## High-Level Purpose
This is a **fact-checking system** built on the DSPy framework that evaluates the factual correctness of statements. The entry point `JudgeModule` is a simplified version that directly judges statements using LLM knowledge without external research, serving as a faster alternative to the full `FactCheckerPipeline` which performs web-based evidence gathering.

## Key Modules and Responsibilities

### 1. **JudgeModule** (Entry Point - Simple Version)
- **Location**: `src/factchecker/simple/modules/judge_module.py`
- **Purpose**: Barebones fact checker that directly evaluates statements using LLM knowledge
- **Components**: Uses `dspy.ChainOfThought` with the `Judge` signature
- **Output**: Returns verdict (SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS), confidence score, and reasoning

### 2. **FactCheckerPipeline** (Full Version)
- **Location**: `src/factchecker/modules/fact_checker_pipeline.py`
- **Purpose**: Complete fact-checking pipeline with iterative web research
- **Sub-modules**:
  - `ClaimExtractorModule`: Breaks statements into atomic verifiable claims
  - `FireJudgeModule`: Evaluates each claim through iterative research cycles
  - `ResearchAgentModule`: Executes web searches and scrapes pages for evidence
  - `AggregatorModule`: Combines claim-level verdicts into overall statement verdict

### 3. **Evaluation & Metrics**
- **Location**: `src/evaluation/evaluate.py`, `src/evaluation/metrics.py`
- **Purpose**: Measures system performance using precision, recall, F1, and accuracy
- **Dataset Schema**: Supports multiple label schemas (FacToolLabelSchema, CSV schema) with normalization

### 4. **Optimization System**
- **Location**: `src/optimizer/gepa_optimize.py`, `src/codeevolver/metric.py`
- **Purpose**: Uses DSPy's GEPA (reflective optimization) to improve fact-checking performance
- **Metric Function**: `gepa_metric` provides score and feedback for optimization

## Data Flow

```
Input Statement → JudgeModule (Simple Path)
                  ├→ Judge Signature (ChainOfThought)
                  └→ Output: verdict, confidence, reasoning

Input Statement → FactCheckerPipeline (Full Path)
                  ├→ ClaimExtractorModule (extract atomic claims)
                  ├→ For each claim:
                  │   └→ FireJudgeModule
                  │       └→ ResearchAgentModule (iterative search + scraping)
                  │           ├→ SerperService (web search)
                  │           └→ FirecrawlService (page scraping)
                  └→ AggregatorModule (combine verdicts)
                      └→ Output: overall_verdict, confidence, reasoning
```

## Metric Being Optimized

The **`gepa_metric`** function optimizes for:
- **Primary**: Correct classification (1.0 score for exact match)
- **Secondary**: Penalizes incorrect predictions (0.0 score) while giving partial credit (0.5) to "UNKNOWN" predictions when evidence is insufficient
- **Focus**: Maximizes **REFUTED class F1 score** and **SUPPORTED class precision** during GEPA optimization
- **Feedback Loop**: Provides textual feedback to the reflection model for iterative improvement

The system uses multi-threaded evaluation with DSPy's evaluation framework, tracking confusion matrices and per-class metrics across SUPPORTED/UNSUPPORTED/REFUTED labels.
