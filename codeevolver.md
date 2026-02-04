PARENT_MODULE_PATH: src.factchecker.modules.judge_module.JudgeModule
METRIC_MODULE_PATH: src.codeevolver.metric.metric

## Architecture Summary

**Purpose**: JudgeModule is a barebones fact-checking system built on DSPy that evaluates statements for factual correctness using LLM knowledge without external research. It serves as a simpler, faster alternative to full fact-checking pipelines.

**Key Modules**:
- **JudgeModule** (`src/factchecker/modules/judge_module.py`): Main entry point, extends `dspy.Module` and uses Chain-of-Thought reasoning to evaluate statements
- **Judge Signature** (`src/factchecker/signatures/judge.py`): DSPy signature defining input/output schema for verdict generation
- **Data Types** (`src/factchecker/models/data_types.py`): Defines structured outputs including JudgmentResult, AggregationResult, and FactCheckResult
- **Metric** (`src/codeevolver/metric.py`): Re-exports `gepa_metric` from the GEPA optimizer for consistency

**Data Flow**:
1. Input statement enters JudgeModule.forward()
2. ChainOfThought predictor processes statement through Judge signature
3. LLM generates reasoning, verdict (SUPPORTED/CONTAINS_UNSUPPORTED_CLAIMS/CONTAINS_REFUTED_CLAIMS), and confidence score
4. Results packaged as dspy.Prediction with all fields

**Metric Being Optimized**: The system uses `gepa_metric` which optimizes for classification accuracy with special handling:
- Score 1.0 for correct predictions
- Score 0.5 for UNKNOWN verdicts (neutral penalty)
- Score 0.0 for incorrect predictions
- Primary focus on maximizing F1 score for REFUTED class detection with secondary emphasis on SUPPORTED precision

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

