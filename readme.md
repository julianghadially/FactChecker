# FactChecker

FactChecker is a DSPy-based fact verification system that assesses the factual correctness of language model outputs. Unlike simple LLM-as-judge approaches that share biases with the models they evaluate, FactChecker grounds its judgments in external evidence through iterative web search. 

## Project Structure

```
src/
├── services/                  # External API integrations
│   ├── serper_service.py      # Google search via Serper
│   └── firecrawl_service.py   # Page scraping via Firecrawl
├── factchecker/
│   ├── signatures/            # DSPy signatures (input/output specs)
│   │   ├── judge.py
│   ├── modules/               # DSPy modules (execution logic)
│   │   └── judge_module.py
│   └── models/                # Data types
│       └── data_types.py
├── baseline/                  # Simple LLM baseline for comparison
│   └── baseline_model.py
├── evaluation/                # Benchmarking system
│   ├── data_loader.py         # HOVER dataset loader
│   ├── metrics.py             # Accuracy, precision calculations
│   └── evaluate.py            # Comparison script
└── main.py                    # CLI entry point
```

## Key Resources
- **DSPy docs**: https://dspy.ai/
- **GEPA Prompt-Optimization paper**: https://arxiv.org/pdf/2507.19457
- **FIRE LLM Fact-Checking paper**: FIRE stands for Fact-checking with Iterative Retrieval and Verification: https://aclanthology.org/2025.findings-naacl.158.pdf?utm_source=chatgpt.com
- **FIRE paper github**: https://github.com/mbzuai-nlp/fire
- **LoCal LLM Fact-checking Paper**: https://dl.acm.org/doi/10.1145/3696410.3714748
 - **Deep Research Dataset**: For future training: https://www.kaggle.com/benchmarks/google/dsqa/leaderboard