```
PARENT_MODULE_PATH: src.factchecker.modules.judge_module.JudgeModule
METRIC_MODULE_PATH: src.codeevolver.metric.metric

## Architecture Summary

**Program Purpose:**
JudgeModule is a fact-checking system that evaluates statements for factual correctness using LLM-based reasoning. It classifies claims as SUPPORTED, CONTAINS_REFUTED_CLAIMS, or CONTAINS_UNSUPPORTED_CLAIMS based on the model's knowledge, without external research or evidence gathering. This is a streamlined alternative to full research-based fact-checking pipelines.

**Key Modules:**
1. **JudgeModule** (src/factchecker/modules/judge_module.py): Main DSPy module that orchestrates fact-checking. Uses ChainOfThought reasoning with the Judge signature.
2. **Judge Signature** (src/factchecker/signatures/judge.py): DSPy signature defining inputs (statement) and outputs (verdict, confidence, reasoning) for the LLM evaluation task.
3. **Metric Function** (src/codeevolver/metric.py): Re-exports gepa_metric for CodeEvolver optimization framework.
4. **GEPA Optimizer** (src/optimizer/gepa_optimize.py): Implements optimization using DSPy's GEPA algorithm with reflective learning.

**Data Flow:**
1. Input statement enters JudgeModule.forward()
2. Judge signature processes via ChainOfThought reasoning, generating verdict with reasoning and confidence
3. Returns dspy.Prediction with overall_verdict, confidence, and reasoning fields
4. During optimization: predictions compared against ground truth via gepa_metric
5. Metric returns score (1.0 correct, 0.5 UNKNOWN, 0.0 incorrect) and feedback for reflection

**Optimization Metric:**
The system optimizes for classification accuracy using gepa_metric, which prioritizes correct verdicts while treating UNKNOWN predictions as neutral (0.5 score). The optimizer uses GEPA's reflective approach to improve prompt strategies and reasoning patterns. Evaluation focuses on REFUTED class F1-score, REFUTED precision/recall, and SUPPORTED precision as key performance indicators.
```

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

