# Architecture Summary: JudgeModule Fact Checker

## High-Level Purpose
The JudgeModule is a lightweight fact-checking system built on DSPy that evaluates the factual correctness of statements. It operates as a simplified alternative to a full fact-checking pipeline, making direct verdicts using LLM knowledge without external research, web searches, or evidence gathering.

## Key Modules and Responsibilities

### 1. **JudgeModule** (Entry Point: `src.factchecker.simple.modules.judge_module.JudgeModule`)
- Core evaluation module that wraps the Judge signature with Chain-of-Thought reasoning
- Takes a statement as input and returns a verdict with confidence score
- Outputs three possible verdicts: `SUPPORTED`, `CONTAINS_UNSUPPORTED_CLAIMS`, or `CONTAINS_REFUTED_CLAIMS`

### 2. **Judge Signature** (`src.factchecker.simple.signatures.judge.py`)
- DSPy signature defining the I/O contract for factual evaluation
- Input: statement to evaluate
- Outputs: reasoning (explanation), verdict (categorical), and confidence (0.0-1.0)

### 3. **Alternative: FactCheckerPipeline** (Full Pipeline)
- More complex pipeline with claim extraction, web research (Serper API), evidence gathering (Firecrawl), iterative judgment, and verdict aggregation
- JudgeModule serves as a faster, simpler alternative when external research isn't needed

## Data Flow

1. **Input**: User provides a statement string
2. **Processing**: JudgeModule invokes ChainOfThought reasoning on the Judge signature
3. **Reasoning**: LLM analyzes the statement based on internal knowledge
4. **Output**: dspy.Prediction containing statement, overall_verdict, confidence, and reasoning

## Metric Being Optimized

**Primary Metric**: `gepa_metric` from `src.optimizer.gepa_optimize.py`

The metric optimizes for:
- **REFUTED class F1 score** (primary target)
- **SUPPORTED class precision** (secondary objective)

Scoring logic:
- Correct prediction: score = 1.0
- UNKNOWN prediction: score = 0.5 (neutral, acceptable when uncertain)
- Incorrect prediction: score = 0.0

The optimizer uses DSPy's GEPA (Generative Evolutionary Prompt Adaptation) to improve fact-checking accuracy through reflective optimization, specifically targeting the identification of false claims while maintaining precision on supported statements.
