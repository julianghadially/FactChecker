# Architecture Summary: Fact-Checking System with LLM Optimization

## High-Level Purpose
This system is an AI-powered fact-checking pipeline that evaluates the factual correctness of statements using large language models (LLMs). The architecture supports two approaches: a **simple judge module** (`JudgeModule`) that makes direct verdicts based on LLM knowledge alone, and a **full pipeline** (`FactCheckerPipeline`) that performs iterative web research to gather evidence before making judgments.

## Key Modules and Responsibilities

### 1. **JudgeModule** (Entry Point)
- **Location**: `src/factchecker/simple/modules/judge_module.py`
- **Purpose**: Barebones fact checker that judges statements directly without external research
- **Components**: Uses DSPy's ChainOfThought reasoning with the `Judge` signature
- **Output**: Returns verdict (SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS), confidence score (0.0-1.0), and reasoning

### 2. **FactCheckerPipeline** (Full Pipeline)
- **Location**: `src/factchecker/modules/fact_checker_pipeline.py`
- **Purpose**: Complete fact-checking with web research and evidence gathering
- **Sub-modules**:
  - **ClaimExtractorModule**: Decomposes statements into atomic claims
  - **FireJudgeModule**: Iteratively evaluates claims with research (max 3 iterations)
  - **ResearchAgentModule**: Performs web searches and scrapes pages (Serper + Firecrawl APIs)
  - **AggregatorModule**: Combines claim-level verdicts into overall statement verdict

### 3. **Judge Signature**
- **Location**: `src/factchecker/simple/signatures/judge.py`
- **Purpose**: Defines input/output schema for LLM-based judgment
- **Fields**: Takes statement as input, outputs reasoning, verdict, and confidence

### 4. **Data Types**
- **Location**: `src/factchecker/models/data_types.py`
- **Purpose**: Type definitions for `JudgmentResult`, `AggregationResult`, and `FactCheckResult`

## Data Flow

### Simple Path (JudgeModule):
1. **Input**: Raw statement string
2. **Processing**: LLM evaluates using ChainOfThought reasoning (no external research)
3. **Output**: Verdict + confidence + reasoning wrapped in `dspy.Prediction`

### Full Pipeline Path:
1. **Input**: Raw statement → ClaimExtractor → List of atomic claims
2. **Per-Claim Loop**: FireJudge → ResearchAgent → Web search/scraping → Evidence → Verdict
3. **Aggregation**: Claim verdicts → Aggregator → Overall statement verdict
4. **Output**: Complete `FactCheckResult` with claim-level details

## Metric Being Optimized

**Metric**: `gepa_metric` (from `src/codeevolver/metric.py`)

**Optimization Goal**: Maximize F1 score for the **REFUTED** class, with secondary focus on **SUPPORTED** class precision

**Scoring Logic**:
- Correct prediction = 1.0 score + positive feedback
- UNKNOWN prediction = 0.5 score (neutral, acceptable when evidence is insufficient)
- Incorrect prediction = 0.0 score + corrective feedback

**Optimizer**: Uses DSPy's GEPA (Generalized Expectation-Based Policy Averaging) with reflective optimization to improve prompts and reasoning chains. The metric provides feedback to GEPA for iterative refinement of the LLM's fact-checking behavior, evaluated on the FacTool QA dataset.
