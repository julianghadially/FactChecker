PARENT_MODULE_PATH: src.factchecker.modules.judge_module.JudgeModule
METRIC_MODULE_PATH: src.codeevolver.metric.metric

## Architecture Summary

**Purpose**: JudgeModule is a lightweight fact-checking system built on DSPy that evaluates statements for factual correctness using LLM reasoning without external research or evidence gathering. It serves as a simplified alternative to full fact-checking pipelines.

**Key Modules**:
- **JudgeModule** (judge_module.py): Core DSPy module that orchestrates fact verification. Wraps a ChainOfThought predictor with the Judge signature to produce verdicts with reasoning.
- **Judge Signature** (signatures/judge.py): DSPy signature defining the input/output schema for statement evaluation. Takes a statement and produces verdict (SUPPORTED/CONTAINS_UNSUPPORTED_CLAIMS/CONTAINS_REFUTED_CLAIMS), confidence score, and reasoning.
- **Data Types** (models/data_types.py): Defines structured outputs including JudgmentResult, AggregationResult, and FactCheckResult for standardized results.

**Data Flow**:
1. Input statement enters JudgeModule.forward()
2. ChainOfThought predictor invokes Judge signature with statement
3. LLM generates reasoning, verdict (3-class), and confidence (0.0-1.0)
4. Results packaged into dspy.Prediction with normalized fields
5. Output flows to evaluation/metrics for F1 score calculation

**Metric Being Optimized**: The gepa_metric function (metric.py) optimizes classification accuracy with special focus on the REFUTED class F1 score. Correct predictions score 1.0, UNKNOWN predictions score 0.5 (neutral), and incorrect predictions score 0.0. The metric provides structured feedback for GEPA's reflective optimization process, aiming to maximize detection of false claims while maintaining precision on supported statements.

## DSPy Patterns and Guidelines

DSPy is an AI framework for defining a compound AI system across multiple modules. Instead of writing prompts, we define signatures. Signatures define the inputs and outputs to a module in an AI system, along with the purpose of the module in the docstring. DSPy leverages a prompt optimizer to convert the signature into an optimized prompt, which is stored as a JSON, and is loaded when compiling the program.

**DSPy docs**: https://dspy.ai/api/

Stick to DSPy for any AI modules you create, unless the client codebase does otherwise.

Defining signatures as classes is recommended. For example:

```python
class WebQueryGenerator(dspy.Signature):
    """Generate a query for searching the web."""
    question: str = dspy.InputField()
    query: str = dspy.OutputField(desc="a query for searching the web")
```

Next, modules are used as nodes in the project, either as a single line:

```python
predict = dspy.Predict(WebQueryGenerator)
```

Or as a class:

```python
class WebQueryModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.query_generator = dspy.Predict(WebQueryGenerator)

    def forward(self, question: str):
        return self.query_generator(question=question)
```

A module can represent a single module, or the module can act as a pipeline that calls a sequence of sub-modules inside `def forward`.

Common prebuilt modules include:
- `dspy.Predict`: for simple language model calls
- `dspy.ChainOfThought`: for reasoning first, followed by a response
- `dspy.ReAct`: for tool calling
- `dspy.ProgramOfThought`: for getting the LM to output code, whose execution results will dictate the response

