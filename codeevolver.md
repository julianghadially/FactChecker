PARENT_MODULE_PATH: src.factchecker.modules.judge_module.JudgeModule
METRIC_MODULE_PATH: src.codeevolver.metric.metric

## Architecture Summary

**Purpose**: This is a fact-checking system built on DSPy that evaluates the factual correctness of statements using LLM-based judgment. The JudgeModule serves as a lightweight fact checker that directly assesses statements without external research, making it faster than full pipeline alternatives that involve web search and evidence gathering.

**Key Modules**:
- **JudgeModule** (`src/factchecker/modules/judge_module.py`): Main entry point. DSPy module that takes a statement and returns a verdict (SUPPORTED/CONTAINS_UNSUPPORTED_CLAIMS/CONTAINS_REFUTED_CLAIMS), confidence score, and reasoning. Uses ChainOfThought prompting with the Judge signature.
- **Judge Signature** (`src/factchecker/signatures/judge.py`): Defines the input/output schema for fact evaluation, specifying three possible verdicts and requiring reasoning + confidence.
- **GEPA Optimizer** (`src/optimizer/gepa_optimize.py`): Implements prompt optimization using DSPy's GEPA (reflective optimization) to maximize REFUTED class F1 score and SUPPORTED class precision. Loads FacTool QA dataset, trains/validates on splits, and evaluates performance.
- **Evaluation Module** (`src/evaluation/metrics.py`): Calculates accuracy, precision, recall, F1 scores, and confusion matrices for fact-checking predictions.

**Data Flow**: 
1. Statement → JudgeModule.forward() 
2. ChainOfThought reasoning with Judge signature 
3. Returns Prediction(overall_verdict, confidence, reasoning)
4. GEPA optimizer uses this pipeline with training data to refine prompts
5. Metric evaluates correctness against ground truth labels

**Metric Being Optimized**: The `gepa_metric` function optimizes for correctness with a scoring scheme: 1.0 for correct predictions, 0.5 for UNKNOWN predictions (neutral), 0.0 for incorrect predictions. Primary focus is maximizing REFUTED F1 score and SUPPORTED precision on the FacTool QA benchmark dataset.

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

