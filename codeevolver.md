PARENT_MODULE_PATH: src.factchecker.modules.judge_module.JudgeModule
METRIC_MODULE_PATH: src.codeevolver.metric.metric

## Architecture Summary

**Purpose**: FactChecker is a DSPy-based fact verification system that evaluates the factual correctness of language model outputs. The JudgeModule serves as a lightweight fact-checking component that assesses statement veracity using LLM knowledge without external research.

**Key Modules**:

1. **JudgeModule** (`src/factchecker/modules/judge_module.py`): Core DSPy module that evaluates statements using ChainOfThought reasoning. Takes a statement as input and produces a verdict (SUPPORTED, CONTAINS_UNSUPPORTED_CLAIMS, or CONTAINS_REFUTED_CLAIMS), confidence score (0.0-1.0), and reasoning explanation.

2. **Judge Signature** (`src/factchecker/signatures/judge.py`): DSPy signature defining the input/output specification for fact evaluation. Specifies statement input field and verdict/reasoning/confidence output fields.

3. **Data Types** (`src/factchecker/models/data_types.py`): Defines structured result types including JudgmentResult, AggregationResult, and FactCheckResult for the broader fact-checking pipeline.

4. **Evaluation System** (`src/evaluation/`): Loads FacTool-QA dataset (true/false labels), calculates metrics (accuracy, precision, recall, F1), and supports multiple label schemas (FacTool, HOVER, ThreeClass).

**Data Flow**: Statement → JudgeModule → ChainOfThought(Judge) → LLM reasoning → Prediction(verdict, confidence, reasoning)

**Metric Being Optimized**: The system optimizes using the `gepa_metric` function which scores predictions based on correctness against ground truth FacTool labels. Correct predictions score 1.0, UNKNOWN predictions score 0.5 (neutral), and incorrect predictions score 0.0. The GEPA optimizer focuses on maximizing REFUTED class F1 score plus SUPPORTED class precision, with feedback for reflective optimization.

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

