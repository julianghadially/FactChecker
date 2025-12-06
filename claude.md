# Claude.md for FactChecker

FactChecker is a project that implements DSPy in Python to build an Agent as a judge fact checker. This agent as a judge is used to assess the factual correctness of the output of another language model. 

This project should be implemented according to the DSPy framework. We are building a "compound AI system" that involves multi hop web search and intermediate language model modules. We are using DSPY so that we can optimize our fact checker. 

Separately, we will need to build another bot (raw language model) that tries to answer questions from the same dataset.

## Key Resources
- **DSPy docs**: https://dspy.ai/
- **HOVER dataset**: https://hover-nlp.github.io/
- **HOVER paper**: [link to arxiv]


## DSPy patterns for all AI modules and Prompts
Prompt engineering is messy. It involves a lot of tweaking, trial and error, and iteration. Additionally, implementing DSPY allows us to optimize a compound AI system across multiple modules

Use DSPy for all AI modules. 

DSPy made up of signatures and modules. Instead of defining prompts, we will define signatures, which defined the inputs and outputs to a module in an AI system. 

Defining signatures as classes is recommended. 
For example:

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
        return self.query_generator(question = question)
```

## Naming conventions
- Use Pascal case for signatures and modules.
- Add Module at the end of all module classes

## HOVER Dataset Structure
- **Format**: JSON lines with `claim`, `supporting_facts`, `label`
- **Location**: `./data/hover/hover_dev.jsonl`
- **Labels**: SUPPORTED, REFUTED, NOT_ENOUGH_INFO

