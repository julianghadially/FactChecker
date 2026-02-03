# Temporal Awareness Module

## Overview

The **TemporalAwarenessModule** is a preprocessing component that detects temporal signals in statements and determines whether they reference events or data beyond the LLM's knowledge cutoff date (June 2024). When temporal uncertainty or post-cutoff dates are detected, the module injects explicit context into the fact-checking pipeline to ensure web searches are performed with temporal-specific queries.

## Problem Solved

Previously, the fact-checking system would default to "CONTAINS_UNSUPPORTED_CLAIMS" for statements about 2025 events or late 2024 data because these were beyond the LLM's knowledge cutoff. The system wasn't proactively recognizing that it needed to search the web with temporal filters to verify such claims.

## Solution

The TemporalAwarenessModule:
1. **Detects temporal signals** (dates, years, phrases like "recently", "this year", etc.)
2. **Determines if events are beyond the knowledge cutoff** (June 2024)
3. **Generates explicit instructions** for the FireJudgeModule to perform web searches
4. **Suggests search strategies** like adding year filters or using news-specific searches

## Architecture

### New Files Created

1. **`src/factchecker/modules/temporal_awareness_module.py`**
   - Contains `TemporalAwarenessModule` class
   - Contains `TemporalAnalysis` DSPy signature for temporal extraction
   - Uses chain-of-thought reasoning to analyze temporal signals

2. **Updated `src/factchecker/models/data_types.py`**
   - Added `TemporalContext` dataclass to store temporal analysis results

### Modified Files

1. **`src/factchecker/modules/fact_checker_pipeline.py`**
   - Added temporal awareness as **Step 1** in the pipeline
   - Passes temporal context to FireJudgeModule when beyond cutoff detected
   - Updated flow documentation

2. **`src/factchecker/modules/fire_judge_module.py`**
   - Added `temporal_context` parameter to `forward()` method
   - Prepends temporal context to evidence string
   - Ensures judge is aware of temporal requirements

## Data Flow

```
Statement Input
      ↓
[1] TemporalAwarenessModule
    - Detects temporal signals
    - Determines if beyond cutoff (June 2024)
    - Generates search instructions
      ↓
[2] ClaimExtractorModule
    - Extracts individual claims
      ↓
[3] FireJudgeModule (for each claim)
    - Receives temporal context if applicable
    - Context prepended to evidence field
    - Instructs judge to perform web searches
    - Suggests temporal query modifiers
      ↓
[4] ResearchAgentModule
    - Executes web searches
    - Can use SerperService.search_news() for recent events
      ↓
[5] AggregatorModule
    - Aggregates claim verdicts
    - Returns overall verdict
```

## TemporalContext Data Structure

```python
@dataclass
class TemporalContext:
    has_temporal_signals: bool          # Whether temporal refs detected
    is_beyond_cutoff: bool              # Whether beyond June 2024
    temporal_entities: list[str]        # Detected dates/phrases
    suggested_search_modifiers: list[str]  # Query modification suggestions
    context_message: str                # Human-readable context for judge
```

## Example Context Message

When a statement contains references beyond the knowledge cutoff, the module generates a context message like:

```
⚠️ TEMPORAL AWARENESS: This claim contains references to events or data
beyond the knowledge cutoff (June 2024).

When searching, prioritize results from 2025.

Consider using SerperService.search_news() to find recent news articles
about this topic with temporal filters (recency='d', 'w', or 'm').

Detected temporal references: 2025, January 20, 2025

🌐 ACTION REQUIRED: You MUST perform web searches to verify this claim.
Do not rely solely on pre-existing knowledge.

Reasoning: The statement explicitly mentions "January 20, 2025" which is
7 months after the June 2024 knowledge cutoff. This requires current web
search to verify.
```

## Usage

### Standalone Usage

```python
from src.factchecker.modules.temporal_awareness_module import TemporalAwarenessModule

# Initialize module
temporal_module = TemporalAwarenessModule()

# Analyze a statement
statement = "The 2025 US presidential inauguration occurred on January 20, 2025."
context = temporal_module(statement=statement)

print(f"Beyond cutoff: {context.is_beyond_cutoff}")
print(f"Temporal entities: {context.temporal_entities}")
print(f"Search modifiers: {context.suggested_search_modifiers}")
```

### Integrated in Pipeline

The module is automatically integrated into the `FactCheckerPipeline`:

```python
from src.factchecker.modules.fact_checker_pipeline import FactCheckerPipeline

# Initialize pipeline (temporal awareness is built-in)
pipeline = FactCheckerPipeline()

# Run fact-checking
result = pipeline(statement="Statement about 2025 events...")

# Temporal context is automatically applied when needed
```

## Key Features

### 1. Temporal Signal Detection
- Extracts dates (e.g., "January 20, 2025")
- Identifies years (e.g., "2025", "2024")
- Recognizes relative phrases (e.g., "recently", "this year", "last month")

### 2. Knowledge Cutoff Awareness
- Compares detected dates against June 2024 cutoff
- Flags high temporal uncertainty for recent/vague references
- Considers current date for relative phrases

### 3. Search Strategy Suggestions
- **Year filters**: Adds specific years to search queries
- **News search**: Recommends `SerperService.search_news()` for recent events
- **Recency parameters**: Suggests 'd' (day), 'w' (week), or 'm' (month) filters

### 4. Explicit Judge Instructions
- Prepends warning to evidence field
- Instructs judge to perform web searches
- Prevents premature "not_supported" verdicts for verifiable claims

## Testing

Run the example script to see the module in action:

```bash
python example_temporal_awareness.py
```

This demonstrates:
1. Temporal signal detection for various statement types
2. Context generation for post-cutoff claims
3. Integration with the full fact-checking pipeline

## Configuration

### Knowledge Cutoff Date

The default cutoff is June 1, 2024, but can be customized:

```python
temporal_module = TemporalAwarenessModule(knowledge_cutoff_date="2024-06-01")
```

### Pipeline Integration

The module is instantiated in `FactCheckerPipeline.__init__()`:

```python
self.temporal_awareness = TemporalAwarenessModule()
```

No additional configuration needed for standard usage.

## Benefits

1. **Improved Accuracy**: Actively searches for recent events instead of defaulting to "unsupported"
2. **Temporal-Aware Queries**: Uses year filters and news search for better results
3. **Explicit Guidance**: Provides clear instructions to the judge module
4. **Modular Design**: Can be used standalone or integrated into pipeline
5. **Configurable**: Knowledge cutoff can be updated as models are updated

## Future Enhancements

Potential improvements:
- Support for date ranges (e.g., "between 2024 and 2025")
- More sophisticated temporal reasoning (e.g., "two months ago")
- Integration with structured data sources for event timelines
- Automatic adjustment of knowledge cutoff based on model version
- Support for non-English temporal phrases

## Dependencies

- **DSPy**: For chain-of-thought temporal analysis
- **datetime**: For date comparisons
- **dataclasses**: For structured context representation

No additional external dependencies required.
