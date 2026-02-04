# Example Usage: Context Fields in Judge Module

## How Context Fields Help

The optional context fields (`topic`, `date`, `source_urls`) provide valuable information to the LLM, helping it make better-informed decisions about factual correctness, especially for:

1. **Time-sensitive claims** - knowing when a statement was made or refers to
2. **Domain-specific claims** - understanding the topic area
3. **Source-contextual claims** - having hints about the reliability or subject matter

## Example 1: Time-Sensitive Claim

### Without Context
```python
judge = JudgeModule()
result = judge(statement="Alaska Airlines will launch flights to London on May 21, 2026.")
# Verdict: CONTAINS_UNSUPPORTED_CLAIMS (can't verify future events)
```

### With Context
```python
judge = JudgeModule()
result = judge(
    statement="Alaska Airlines will launch flights to London on May 21, 2026.",
    topic="Alaska Air",
    date="20251210",  # This claim was made on Dec 10, 2025
    source_urls="https://www.cbsnews.com/news/alaska-airlines-london-flights/"
)
# The LLM can now understand this is about a future announcement and has temporal context
# about when the claim was made relative to the event date
```

## Example 2: Domain-Specific Claim

### Without Context
```python
result = judge(statement="The new quantum processor achieved 99.9% fidelity.")
# Verdict: Depends on LLM's general knowledge, may be uncertain
```

### With Context
```python
result = judge(
    statement="The new quantum processor achieved 99.9% fidelity.",
    topic="Quantum Computing",
    date="20250115",
    source_urls="https://arxiv.org/quantum-processor-paper"
)
# The LLM knows this is in the quantum computing domain and can apply
# domain-specific reasoning about what "fidelity" means in this context
```

## Example 3: Dataset Integration

When using datasets with context fields (like `FactChecker_news_claims.csv`):

```python
from src.optimizer.gepa_optimize import load_dspy_examples

# Load examples - context fields are automatically extracted
examples = load_dspy_examples("data/FactChecker_news_claims.csv")

# Each example now includes:
# - statement: the claim to verify
# - label: ground truth (SUPPORTED/REFUTED)
# - topic: domain context (e.g., "Alaska Air", "Politics")
# - date: when the claim was generated (YYYYMMDD)
# - source_urls: relevant source URLs

# These are passed to the judge during training and evaluation
judge = JudgeModule()
for example in examples:
    result = judge(
        statement=example.statement,
        topic=example.topic,
        date=example.date,
        source_urls=example.source_urls
    )
```

## Example 4: GEPA Optimization with Context

The GEPA optimizer automatically uses context fields during training:

```python
# Run optimization with a dataset that has context fields
python -m src.optimizer.gepa_optimize \
    --auto light \
    --model openai/gpt-5-mini \
    --mlflow

# The optimizer will:
# 1. Load examples with context fields from the dataset
# 2. Pass context to the judge during training
# 3. Learn to use topic, date, and source_urls to make better predictions
# 4. Optimize prompts that incorporate this contextual information
```

## Backward Compatibility

The system remains fully backward compatible. If your dataset doesn't have context fields:

```python
# Old code still works - empty strings are used as defaults
examples = load_dspy_examples("data/FacTool_QA_train.jsonl")
# topic="", date="", source_urls="" for all examples

judge = JudgeModule()
result = judge(statement="Some claim to verify")
# Works exactly as before - context fields default to ""
```

## Dataset Format Requirements

### CSV Format (with context)
```csv
topic,claim,label,url,date_generated,Reviewed
Alaska Air,"Alaska Airlines will launch flights to London",TRUE,https://example.com,20251210,claim_only
Politics,"The bill was passed in Congress",FALSE,https://gov.example.com,20250115,claim_only
```

### JSONL Format (with context)
```jsonl
{"claim": "Some claim", "label": "true", "topic": "Technology", "date_generated": "20250101", "url": "https://example.com"}
{"claim": "Another claim", "label": "false", "topic": "Science", "date_generated": "20250102", "url": "https://example2.com"}
```

### JSONL Format (without context - backward compatible)
```jsonl
{"claim": "Some claim", "label": "true"}
{"claim": "Another claim", "label": "false"}
```

## Benefits Summary

1. **Better Temporal Reasoning**: LLM can understand time-sensitive claims
2. **Domain Awareness**: Topic field helps with specialized knowledge areas
3. **Source Hints**: URLs provide context about claim origin
4. **No API Calls**: All context comes from the dataset, no external APIs needed
5. **Improved Training**: GEPA optimizer can learn to use contextual information
6. **Backward Compatible**: Works with or without context fields
