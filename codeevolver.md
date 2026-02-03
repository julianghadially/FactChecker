# Architecture Summary: JudgeModule Fact-Checking System

## High-Level Purpose
This system performs automated fact-checking of statements using LLM-based reasoning. The **JudgeModule** serves as a lightweight, barebones fact-checker that evaluates statements directly without external research, providing verdicts on factual correctness along with confidence scores and reasoning.

## Key Modules and Responsibilities

### 1. **JudgeModule** (Entry Point: `src/factchecker/simple/modules/judge_module.py`)
- **Role**: Simplified fact-checker that judges statements using LLM knowledge alone
- **Implementation**: Uses DSPy's `ChainOfThought` reasoning with the `Judge` signature
- **Input**: Single statement string
- **Output**: Verdict (SUPPORTED/CONTAINS_UNSUPPORTED_CLAIMS/CONTAINS_REFUTED_CLAIMS), confidence score (0-1), and reasoning explanation
- **Distinguishing feature**: No claim extraction, no web search, no evidence gathering—pure LLM reasoning

### 2. **Judge Signature** (`src/factchecker/simple/signatures/judge.py`)
- Defines the DSPy signature for fact evaluation
- Enforces structured output with three verdict categories and confidence scoring
- Prompts the LLM to assess factual correctness based on internal knowledge

### 3. **Alternative: FactCheckerPipeline** (`src/factchecker/modules/fact_checker_pipeline.py`)
- Full-featured fact-checking pipeline (not the optimized entry point, but related)
- Multi-stage flow: claim extraction → iterative web research → per-claim judgment → aggregation
- Components: ClaimExtractorModule, ResearchAgentModule, FireJudgeModule, AggregatorModule
- Uses external services (Serper search, Firecrawl scraping) for evidence gathering

## Data Flow

```
Statement Input
    ↓
JudgeModule.forward()
    ↓
DSPy ChainOfThought(Judge signature)
    ↓
LLM evaluates statement factuality
    ↓
dspy.Prediction(
    overall_verdict,
    confidence,
    reasoning
)
```

## Metric Being Optimized

**Primary Metric**: `gepa_metric` (from `src/codeevolver/metric.py`)

The metric optimizes for **correctness of verdicts** with a tiered scoring system:
- **Score 1.0**: Prediction matches ground truth label exactly
- **Score 0.5**: Prediction is "UNKNOWN" (neutral penalty for uncertainty)
- **Score 0.0**: Incorrect prediction

**Key Optimization Goals** (from GEPA optimizer):
1. **REFUTED F1 score**: Maximize F1 for detecting false claims (primary focus)
2. **SUPPORTED precision**: High precision on supported claims (secondary goal)
3. Balance between detecting misinformation (REFUTED) and avoiding false positives

The GEPA (Generalized Expectation Prompting with Aggregation) optimizer uses this metric with reflective feedback to iteratively improve the model's prompt engineering and reasoning strategies, aiming to improve fact-checking accuracy across train/validation/test datasets.
