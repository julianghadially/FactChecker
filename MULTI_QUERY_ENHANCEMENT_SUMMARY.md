# Multi-Query Search Enhancement Summary

## Overview

Enhanced the `_gather_evidence` method in `src/factchecker/simple/modules/judge_module.py` to implement a multi-query search strategy. This improvement enables the fact-checker to retrieve more targeted evidence for temporal and numeric claims, leading to more accurate verdicts (especially REFUTED instead of CONTAINS_UNSUPPORTED_CLAIMS).

## Changes Made

### 1. New DSPy Signature: QueryGenerator

**File:** `src/factchecker/simple/signatures/query_generator.py`

Created a new DSPy signature that extracts 1-3 focused search queries from a statement:

```python
class QueryGenerator(Signature):
    """Extract focused search queries from a statement to gather comprehensive evidence.

    Analyze the statement and generate 1-3 specific, targeted search queries that would
    help verify different aspects of the claim, especially temporal and numeric details.
    """

    statement: str = InputField(desc="The statement to generate search queries for")
    queries: list[str] = OutputField(
        desc="1-3 focused search queries that target different aspects of the statement, "
             "especially temporal and numeric claims. Each query should be concise (5-15 words)."
    )
```

**Example Usage:**
- **Input:** "Mondelez has been selling sugar-free Oreo cookies in the United States for several years prior to the announced Oreo Zero Sugar launch"
- **Output:**
  1. "Oreo Zero Sugar launch date"
  2. "sugar-free Oreo United States history before 2026"
  3. "Mondelez sugar-free Oreo products US availability"

### 2. Enhanced JudgeModule

**File:** `src/factchecker/simple/modules/judge_module.py`

#### Updated `__init__` method:
- Added `self.query_generator = dspy.ChainOfThought(QueryGenerator)` to initialize the query generation module

#### Redesigned `_gather_evidence` method:

**Previous implementation:**
- Executed ONE search query using the full statement
- Scraped top 2 results
- Limited evidence coverage

**New implementation:**
```python
def _gather_evidence(self, statement: str) -> str:
    """Perform lightweight web research using multi-query search strategy.

    Uses QueryGenerator to extract 1-3 focused search queries from the statement,
    executes all queries, and collects up to 3-4 deduplicated sources.
    """
    # Step 1: Generate focused search queries
    query_result = self.query_generator(statement=statement)
    queries = query_result.queries[:3]  # Limit to max 3 queries

    # Step 2: Execute all queries and collect results
    # - Deduplicates by URL
    # - Collects up to 4 total sources

    # Step 3: Scrape top 3-4 results
    # - Returns formatted evidence
```

**Key improvements:**
1. **Multi-query approach:** Generates multiple targeted queries instead of one broad query
2. **Better coverage:** Up to 4 sources (increased from 2)
3. **Deduplication:** Prevents duplicate URLs across different queries
4. **Fallback handling:** Uses original statement if query generation fails

### 3. Updated Module Documentation

Updated the JudgeModule docstring to reflect the new multi-query strategy:

```python
"""
When triggered, performs:
1. Generates 1-3 focused search queries using QueryGenerator
2. Executes all queries and collects results
3. Scrapes top results (3-4 sources total, deduplicated by URL)
4. Re-evaluates with aggregated evidence passed to Judge

This multi-query approach helps retrieve evidence that directly addresses
specific temporal and numeric claims, enabling more accurate verdicts.
"""
```

### 4. Updated Exports

**File:** `src/factchecker/simple/signatures/__init__.py`

Added QueryGenerator to the module exports:
```python
from src.factchecker.simple.signatures.query_generator import QueryGenerator
__all__ = ["Judge", "QueryGenerator"]
```

## Benefits

### 1. More Accurate Verdicts
- **Before:** Would often return CONTAINS_UNSUPPORTED_CLAIMS when evidence existed but wasn't found with a single broad query
- **After:** Can return REFUTED verdicts when finding contradictory evidence through targeted queries

### 2. Better Temporal & Numeric Claim Handling
- Queries specifically target dates, numbers, and temporal relationships
- Example: For "several years prior to launch", generates queries about both the launch date AND historical availability

### 3. Improved Evidence Quality
- Multiple focused queries retrieve more relevant sources
- Deduplication ensures diverse perspectives
- Up to 4 sources provide comprehensive coverage

### 4. Maintained Efficiency
- Still lightweight (1-3 queries, 3-4 sources)
- Falls back gracefully if query generation fails
- Existing trigger conditions unchanged

## Example Workflow

**Statement:** "Mondelez has been selling sugar-free Oreo cookies in the United States for several years prior to the announced Oreo Zero Sugar launch"

### Step 1: First Pass (LLM Knowledge Only)
- Verdict: CONTAINS_UNSUPPORTED_CLAIMS
- Confidence: 0.4
- Reasoning: "Cannot verify the timeline of sugar-free Oreo availability before 2024 knowledge cutoff"

### Step 2: Research Triggered
Research trigger condition met (confidence < 0.6)

### Step 3: Query Generation
QueryGenerator produces:
1. "Oreo Zero Sugar launch date"
2. "sugar-free Oreo United States history before 2026"
3. "Mondelez sugar-free Oreo products US availability"

### Step 4: Multi-Query Search
- Execute all 3 queries
- Collect deduplicated results
- Scrape top 4 sources

### Step 5: Second Pass (With Evidence)
- Verdict: REFUTED (if evidence contradicts) or SUPPORTED (if evidence confirms)
- Confidence: Higher (due to evidence)
- Reasoning: Cites specific evidence from sources

## Testing

A test script has been created: `test_multi_query_enhancement.py`

Tests verify:
1. QueryGenerator creates focused search queries
2. JudgeModule properly integrates the query_generator
3. All components work together

## Files Modified

1. `src/factchecker/simple/signatures/query_generator.py` (NEW)
2. `src/factchecker/simple/signatures/__init__.py` (MODIFIED)
3. `src/factchecker/simple/modules/judge_module.py` (MODIFIED)

## Backward Compatibility

✓ Fully backward compatible
- Existing trigger conditions unchanged
- Forward method signature unchanged
- Returns same output format
- Falls back to original statement if query generation fails

## Future Enhancements

Potential improvements:
1. Add query generation reasoning to output for transparency
2. Tune number of queries based on statement complexity
3. Add query quality scoring
4. Implement query caching for similar statements
