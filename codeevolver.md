PARENT_MODULE_PATH: src.factchecker.modules.judge_module.JudgeModule
METRIC_MODULE_PATH: src.codeevolver.metric.metric

## Architecture Summary

**Purpose**: This is a lightweight fact-checking system that evaluates statements for factual correctness using LLM-based reasoning without external research or evidence gathering.

**Key Modules**:
- **JudgeModule** (`src.factchecker.modules.judge_module`): The main entry point that wraps a DSPy ChainOfThought predictor to evaluate statements directly using the LLM's internal knowledge. Returns verdicts (SUPPORTED, CONTAINS_UNSUPPORTED_CLAIMS, CONTAINS_REFUTED_CLAIMS) with confidence scores and reasoning.
- **Judge Signature** (`src.factchecker.signatures.judge`): DSPy signature defining the input/output contract for the judge predictor - takes a statement and outputs verdict, reasoning, and confidence.
- **Evaluation System** (`src.evaluation.metrics`): Calculates accuracy, precision, recall, and F1 scores per class using a confusion matrix approach on normalized predictions.

**Data Flow**: 
1. Input statement → JudgeModule.forward()
2. Statement passed to ChainOfThought(Judge) predictor
3. LLM generates verdict, confidence, and reasoning
4. Results wrapped in dspy.Prediction and returned

**Optimization Metric**: The `gepa_metric` function provides binary feedback (score 1.0 for correct, 0.5 for neutral UNKNOWN predictions, 0.0 for incorrect) optimizing classification accuracy. The system focuses on maximizing REFUTED class F1 score and SUPPORTED class precision, critical for fact-checking applications where false positives/negatives have high costs.

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

