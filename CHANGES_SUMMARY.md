# Summary of Changes: Optional Context Fields for Judge Module

## Overview
Modified the Judge signature and related modules to accept optional context fields (topic, date, source_urls) that provide additional context to help the LLM make more informed verdicts about time-sensitive or domain-specific claims without requiring external API calls.

## Files Modified

### 1. `src/factchecker/signatures/judge.py`
**Changes:**
- Added three optional `InputField` parameters to the `Judge` signature:
  - `topic`: Optional context about the topic or domain (e.g., 'Alaska Air', 'Politics')
  - `date`: Optional context about when the statement was generated or refers to (YYYYMMDD format)
  - `source_urls`: Optional comma-separated URLs that provide relevant context

**Implementation:**
```python
topic: str = InputField(
    default="",
    desc="Optional context: The topic or domain this statement relates to (e.g., 'Alaska Air', 'Politics')"
)
date: str = InputField(
    default="",
    desc="Optional context: The date when this statement was generated or refers to (YYYYMMDD format)"
)
source_urls: str = InputField(
    default="",
    desc="Optional context: Comma-separated URLs that provide relevant context for this statement"
)
```

### 2. `src/factchecker/modules/judge_module.py`
**Changes:**
- Updated `JudgeModule.forward()` method to accept optional context parameters
- Modified the method to pass these parameters to the judge signature

**Implementation:**
```python
def forward(
    self,
    statement: str,
    topic: str = "",
    date: str = "",
    source_urls: str = ""
) -> dspy.Prediction:
    """Evaluate a statement for factual correctness.

    Args:
        statement: The statement to evaluate.
        topic: Optional context about the topic or domain.
        date: Optional context about when the statement was generated (YYYYMMDD format).
        source_urls: Optional comma-separated URLs providing relevant context.

    Returns:
        dspy.Prediction with verdict, confidence, and reasoning.
    """
    result = self.judge(
        statement=statement,
        topic=topic,
        date=date,
        source_urls=source_urls
    )
    # ... rest of implementation
```

### 3. `src/evaluation/data_loader.py`
**Changes:**
- Added optional context fields to the `HoverExample` dataclass:
  - `topic: str = ""`
  - `date_generated: str = ""`
  - `url: str = ""`
- Updated `load_dataset()` function to extract these fields from dataset items
- Updated `load_csv_dataset()` function to read and include these fields from CSV files

**Key Changes:**
- Modified the CSV reader to capture `date_generated` field
- Updated example creation in both `load_dataset()` and `load_csv_dataset()` to include context fields

### 4. `src/optimizer/gepa_optimize.py`
**Changes:**
- Updated `load_dspy_examples()` to include context fields when creating `dspy.Example` objects:
  - Maps `ex.topic` → `topic`
  - Maps `ex.date_generated` → `date`
  - Maps `ex.url` → `source_urls`
- Added support for CSV files by importing and using `load_csv_dataset` when path ends with `.csv`
- Modified `evaluate_program()` to pass context fields when calling the program

**Implementation:**
```python
def load_dspy_examples(path: str, limit: Optional[int] = None) -> list[dspy.Example]:
    """Load dataset as DSPy Examples with optional context fields."""
    from src.evaluation.data_loader import load_csv_dataset

    # Determine loader based on file extension
    if path.endswith('.csv'):
        dataset = load_csv_dataset(path=path, limit=limit)
    else:
        dataset = load_dataset(path=path, limit=limit)

    examples = []
    for ex in dataset.examples:
        normalized_label = FacToolLabelSchema.normalize_ground_truth(ex.label)
        examples.append(
            dspy.Example(
                statement=ex.claim,
                label=normalized_label,
                topic=ex.topic,
                date=ex.date_generated,
                source_urls=ex.url
            ).with_inputs("statement", "topic", "date", "source_urls")
        )
    return examples
```

### 5. `src/evaluation/evaluate.py`
**Changes:**
- Updated the creation of `dspy.Example` objects to include context fields using `getattr()` for safe attribute access

**Implementation:**
```python
examples = [
    dspy.Example(
        statement=ex.claim,
        label=ex.label,
        topic=getattr(ex, 'topic', ''),
        date=getattr(ex, 'date_generated', ''),
        source_urls=getattr(ex, 'url', '')
    ).with_inputs("statement", "topic", "date", "source_urls")
    for ex in dataset
]
```

## Benefits

1. **Better Context for LLM**: The LLM can now use domain-specific and temporal context to make more informed decisions
2. **Time-Sensitive Claims**: Date information helps the LLM understand if a claim is about future events or past events
3. **Domain Awareness**: Topic field provides domain context (e.g., "Alaska Air", "Politics") to help with specialized knowledge
4. **Source Context**: URLs can provide hints about the reliability or topic area of claims
5. **Backward Compatible**: All context fields are optional with empty string defaults, so existing code continues to work
6. **No External APIs Required**: Context is provided directly from the dataset, avoiding need for external API calls

## Testing

Created comprehensive test files:

**`test_context_fields.py`** - Basic unit tests:
1. Module works without context fields (backward compatibility)
2. Module correctly accepts and uses context fields
3. Integration with dspy.Example works as expected

**`test_integration.py`** - End-to-end integration tests:
1. Data loader extracts context fields from CSV datasets
2. DSPy example creation includes context fields as inputs
3. JudgeModule processes context fields correctly
4. Backward compatibility with datasets lacking context fields

All tests pass successfully ✓

## Dataset Compatibility

- **FacTool_QA datasets**: Don't have context fields - will use empty strings (backward compatible)
- **FactChecker_news_claims.csv**: Has `topic`, `date_generated`, and `url` fields - will be fully utilized
- **Other datasets**: Will gracefully handle presence or absence of context fields using `getattr()` with defaults

## Next Steps

To use these context fields effectively:
1. Ensure your dataset includes `topic`, `date_generated`, and/or `url` fields
2. The GEPA optimizer will automatically include these fields during training
3. The LLM will learn to use this context to make better predictions on time-sensitive or domain-specific claims
