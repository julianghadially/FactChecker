# JudgeModule Enhancement: Web Search Fallback

## Overview

The `JudgeModule` has been enhanced with a **hybrid two-stage architecture** that intelligently combines LLM knowledge with web search to handle claims beyond the model's knowledge cutoff.

## Problem Solved

The original `JudgeModule` could only rely on the LLM's training data, which has a knowledge cutoff date. This meant:
- ❌ Recent events (2024-2025) couldn't be verified
- ❌ Statements about current news resulted in "CONTAINS_UNSUPPORTED_CLAIMS"
- ❌ No way to verify time-sensitive information

## Solution: Hybrid Two-Stage Pipeline

### Stage 1: LLM-Only Judgment (Fast Path)
The module first attempts to evaluate the statement using only the LLM's knowledge:
```python
result = self.judge(statement=statement)
```

**Benefits:**
- ⚡ Fast - no external API calls
- 💰 Cost-effective - no web scraping costs
- ✅ Works great for historical facts and well-known information

### Stage 2: Web-Augmented Judgment (Fallback)
If the LLM is uncertain, the module automatically:

1. **Detects Uncertainty** via two triggers:
   - Confidence score < 0.7
   - Reasoning contains uncertainty indicators:
     - "knowledge cutoff"
     - "lacking information"
     - "unable to verify"
     - "cannot confirm"
     - "don't have access"
     - "no current/recent information"
     - "beyond my training"
     - "needs more recent data"

2. **Gathers Web Evidence**:
   ```python
   search_results = self.serper.search(query=statement, num_results=3)
   ```
   - Performs Google search via SerperService
   - Scrapes 2-3 top results with FirecrawlService
   - Extracts both snippets and full page content (up to 3000 chars each)

3. **Re-evaluates with Evidence**:
   ```python
   web_result = self.web_judge(statement=statement, evidence=evidence)
   ```
   - Uses `WebAugmentedJudge` signature
   - Makes informed decision based on scraped evidence
   - Returns updated verdict and confidence

## Architecture Changes

### New Files Created

1. **`src/factchecker/simple/signatures/web_augmented_judge.py`**
   - New DSPy signature for evidence-based judging
   - Takes statement + evidence as input
   - Returns verdict, confidence, and reasoning

### Modified Files

1. **`src/factchecker/simple/modules/judge_module.py`**
   - Added imports: `SerperService`, `FirecrawlService`, `WebAugmentedJudge`, `re`
   - Enhanced `__init__()` to initialize web services
   - Updated `forward()` method with two-stage logic
   - Added `_gather_web_evidence()` helper method
   - New parameter: `web_search_enabled` (default: `True`)

2. **`src/factchecker/simple/signatures/__init__.py`**
   - Exported `WebAugmentedJudge` signature

## Usage

### Basic Usage (with web search enabled by default)
```python
from src.factchecker.simple.modules.judge_module import JudgeModule

judge = JudgeModule()
result = judge.forward("Donald Trump won the 2024 U.S. presidential election")

print(f"Verdict: {result.overall_verdict}")
print(f"Confidence: {result.confidence}")
print(f"Used web search: {result.used_web_search}")
print(f"Reasoning: {result.reasoning}")
```

### Disable Web Search (original behavior)
```python
result = judge.forward(statement, web_search_enabled=False)
```

### Output Schema
```python
dspy.Prediction(
    statement: str,              # Input statement
    overall_verdict: str,        # SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
    confidence: float,           # 0.0 to 1.0
    reasoning: str,              # Explanation of verdict
    used_web_search: bool,       # True if web search was triggered
    evidence: str | None,        # Retrieved evidence (only if web search used)
)
```

## Performance Characteristics

### Fast Path (LLM-Only)
- **Latency**: ~1-3 seconds
- **Cost**: 1 LLM API call
- **Triggers**: High confidence (≥0.7) + no uncertainty indicators

### Slow Path (Web-Augmented)
- **Latency**: ~10-20 seconds
- **Cost**: 2 LLM API calls + 1 search + 2-3 scrapes
- **Triggers**: Low confidence (<0.7) OR uncertainty indicators

## Example Scenarios

### Scenario 1: Historical Fact (Fast Path)
**Statement**: "The United States declared independence in 1776"

**Flow**:
1. LLM judges → High confidence (0.95)
2. No uncertainty indicators
3. ✅ Returns LLM-only verdict (SUPPORTED)
4. ⚡ No web search needed

### Scenario 2: Recent Event (Slow Path)
**Statement**: "Donald Trump won the 2024 U.S. presidential election"

**Flow**:
1. LLM judges → Low confidence (0.3)
2. Reasoning mentions "knowledge cutoff"
3. 🌐 Triggers web search
4. Scrapes news articles about 2024 election
5. Re-evaluates with evidence
6. ✅ Returns evidence-based verdict (SUPPORTED)

### Scenario 3: False Recent Claim (Slow Path)
**Statement**: "OpenAI released GPT-5 in 2025"

**Flow**:
1. LLM judges → Low confidence (0.4)
2. Reasoning mentions "lacking information"
3. 🌐 Triggers web search
4. Scrapes OpenAI announcements and tech news
5. Evidence shows no GPT-5 release
6. ✅ Returns verdict (CONTAINS_REFUTED_CLAIMS)

## Testing

Run the test script to see the enhancement in action:
```bash
python test_judge_enhancement.py
```

This will test:
- Recent events requiring web search
- Historical facts NOT requiring web search
- False claims about recent events

## Advantages

1. **✅ Solves Knowledge Cutoff Problem**
   - Can now verify recent events and current information

2. **⚡ Maintains Speed for Known Facts**
   - Historical/well-known facts bypass web search entirely

3. **🎯 Smart Triggering Logic**
   - Two independent triggers (confidence + reasoning analysis)
   - Catches both quantitative and qualitative uncertainty

4. **💪 Robust Evidence Gathering**
   - Multiple sources (2-3 top results)
   - Full page scraping (not just snippets)
   - Formatted evidence for LLM consumption

5. **🔌 Backward Compatible**
   - `web_search_enabled=False` preserves original behavior
   - Existing code works without changes

6. **📊 Transparent Decision Making**
   - `used_web_search` flag shows which path was taken
   - `evidence` field provides full retrieved context

## Implementation Details

### Uncertainty Detection Patterns
```python
uncertainty_patterns = [
    r"knowledge cutoff",
    r"cutoff date",
    r"lack(?:ing)?\s+(?:sufficient\s+)?information",
    r"unable to verify",
    r"cannot (?:confirm|verify)",
    r"don't have (?:access|information)",
    r"no (?:current|recent|up-to-date) information",
    r"(?:as of|beyond) my (?:knowledge|training)",
    r"need(?:s)? more (?:recent|current|up-to-date) (?:information|data)",
]
```

### Evidence Formatting
```
=== WEB SEARCH RESULTS ===

--- Source 1: Title ---
URL: https://...
Snippet: Preview text...

Full Content (truncated):
[Scraped markdown content up to 3000 chars]

--- Source 2: Title ---
...
```

## Configuration

### Adjustable Parameters

In `_gather_web_evidence()`:
- `num_results`: Number of search results to scrape (default: 3)
- `max_length`: Max chars per scraped page (default: 3000)

In `forward()`:
- Confidence threshold: Currently hardcoded to 0.7, can be parameterized

## Future Enhancements

Potential improvements:
1. Make confidence threshold configurable
2. Add caching for repeated statements
3. Support for news-specific search (SerperService.search_news)
4. Parallel scraping for faster evidence gathering
5. Evidence quality scoring
6. Source credibility ranking

## Dependencies

- `dspy`: For signatures and modules
- `SerperService`: Google search API
- `FirecrawlService`: Web scraping service
- `re`: Regular expressions for pattern matching

## Summary

The enhanced `JudgeModule` provides a best-of-both-worlds solution:
- **Fast** for facts the LLM knows
- **Accurate** for recent events via web search
- **Intelligent** fallback triggering
- **Transparent** decision process
- **Backward compatible** with existing code

This creates a production-ready fact-checking module that handles the full spectrum of claims, from historical facts to breaking news.
