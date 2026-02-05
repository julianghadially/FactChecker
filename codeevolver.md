PARENT_MODULE_PATH: src.factchecker.modules.judge_module.JudgeModule
METRIC_MODULE_PATH: src.codeevolver.metric.metric

## Architecture Summary

**Purpose**: This is a fact-checking system built with DSPy that evaluates statements for factual correctness. The JudgeModule serves as a lightweight fact-checker that directly assesses statements using LLM knowledge without external research or evidence gathering.

**Key Modules**:
- **JudgeModule** (`src/factchecker/modules/judge_module.py`): The main evaluation module that extends dspy.Module. It uses ChainOfThought reasoning to judge statements and returns verdicts: SUPPORTED, CONTAINS_UNSUPPORTED_CLAIMS, or CONTAINS_REFUTED_CLAIMS, along with confidence scores and reasoning.
- **Judge Signature** (`src/factchecker/signatures/judge.py`): DSPy signature defining the input/output schema for fact evaluation. Takes a statement string and outputs reasoning, verdict (categorical), and confidence (float 0.0-1.0).
- **GEPA Metric** (`src/codeevolver/metric.py` → `src/optimizer/gepa_optimize.py`): The optimization metric that scores predictions based on correctness. Returns score=1.0 for correct predictions, 0.5 for neutral UNKNOWN predictions, and 0.0 for incorrect predictions, with explanatory feedback for reflective optimization.

**Data Flow**: Statement → JudgeModule.forward() → Judge signature with ChainOfThought → LLM reasoning → dspy.Prediction with verdict, confidence, and reasoning → GEPA metric evaluates against ground truth.

**Metric Being Optimized**: The GEPA metric optimizes for classification accuracy with three classes (SUPPORTED/REFUTED/UNKNOWN), where correct predictions score 1.0, UNKNOWN predictions receive partial credit (0.5), and incorrect predictions score 0.0. The broader goal is maximizing F1 score for the REFUTED class and precision for SUPPORTED class on the FacTool QA dataset.

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

