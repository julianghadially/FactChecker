# FactChecker Simple JudgeModule - Architecture Summary

## High-Level Purpose

The `JudgeModule` is a streamlined fact-checking system that evaluates factual statements directly using LLM knowledge without external research. It serves as a lightweight alternative to the full FactChecker pipeline, trading comprehensive evidence gathering for speed and simplicity. The module takes textual statements as input and outputs verdicts (SUPPORTED, CONTAINS_UNSUPPORTED_CLAIMS, or CONTAINS_REFUTED_CLAIMS) with confidence scores and reasoning.

## Key Modules and Responsibilities

### 1. **JudgeModule** (`src/factchecker/simple/modules/judge_module.py`)
- **Entry point** for the fact-checking system
- Wraps a DSPy ChainOfThought predictor around the Judge signature
- Transforms input statements into structured predictions with verdicts, confidence scores, and reasoning
- No external API calls or web searches - relies solely on LLM's parametric knowledge

### 2. **Judge Signature** (`src/factchecker/simple/signatures/judge.py`)
- Defines the input/output contract for the fact-checking task
- Input: A statement string to evaluate
- Outputs: 
  - `reasoning`: Explanation for the verdict
  - `verdict`: One of three categorical labels (SUPPORTED, CONTAINS_UNSUPPORTED_CLAIMS, CONTAINS_REFUTED_CLAIMS)
  - `confidence`: Float score between 0.0 and 1.0

### 3. **Metric Function** (`src/codeevolver/metric.py`)
- Re-exports `gepa_metric` from the GEPA optimizer
- Provides feedback for DSPy's GEPA (Generalized Error-driven Prompt Adaptation) optimization
- Scoring logic:
  - Correct predictions: score = 1.0
  - UNKNOWN predictions: score = 0.5 (neutral penalty)
  - Incorrect predictions: score = 0.0

## Data Flow

```
Input Statement
    ↓
JudgeModule.__init__()
    → Creates DSPy ChainOfThought wrapper around Judge signature
    ↓
JudgeModule.forward(statement)
    → LLM evaluates statement using chain-of-thought reasoning
    → Produces: verdict, confidence, reasoning
    ↓
dspy.Prediction(statement, overall_verdict, confidence, reasoning)
    ↓
Output: Structured prediction object
```

## Metric Being Optimized

**Primary Metric**: **REFUTED F1 Score** - The harmonic mean of precision and recall for identifying false/refuted claims.

**Secondary Considerations**:
- SUPPORTED precision (accuracy of true claim identification)
- Overall accuracy on predicted cases
- The metric uses a three-way penalty structure that encourages the model to be confident when certain, but allows UNKNOWN predictions when evidence is insufficient (50% partial credit)

**Optimization Approach**: GEPA (Generalized Error-driven Prompt Adaptation) iteratively refines prompts through reflection on prediction errors, using feedback from the metric to guide improvements.

**Context**: This simple JudgeModule serves as a baseline for comparison against the full FactChecker pipeline, which includes claim extraction, iterative web research, and evidence aggregation. The full pipeline achieves 96% accuracy with 83% recall on refuted claims, while this simple version trades accuracy for speed by eliminating external research.
