# Query Generator Implementation - Intelligent Web Search Enhancement

## Overview

This implementation adds an intelligent query generation module to improve web search effectiveness in the fact-checking system. Instead of searching for verbatim statements, the system now generates optimized, domain-aware search queries that extract key entities and formulate targeted queries for better evidence retrieval.

## Changes Made

### 1. New File: `src/factchecker/simple/signatures/query_generator.py`

**Purpose**: DSPy signature that takes a statement as input and outputs 1-3 optimized search queries.

**Key Features**:
- Extracts key entities (companies, people, dates, specific claims)
- Formulates domain-specific queries (e.g., "Deutsche Bank 3M rating 2025" instead of full statement)
- Targets authoritative sources
- Breaks complex statements into focused searchable components

**Example Transformations**:
```python
# Input: "Deutsche Bank upgraded 3M to buy"
# Output: ["Deutsche Bank 3M rating 2025", "3M stock upgrade Deutsche Bank"]

# Input: "Apple released iPhone 15 in September 2023"
# Output: ["Apple iPhone 15 release date", "iPhone 15 launch September 2023"]
```

### 2. Modified: `src/factchecker/simple/modules/judge_module.py`

**Changes to `__init__` method**:
- Added `self.query_generator = dspy.ChainOfThought(QueryGenerator)` to initialize the query generation module
- Added import for `QueryGenerator`

**Changes to `_gather_web_evidence` method**:

The method now follows a 3-step process:

#### Step 1: Generate Optimized Queries
```python
query_result = self.query_generator(statement=statement)
queries = query_result.queries[:3]  # Limit to 3 queries max
```

#### Step 2: Execute Searches & Aggregate Results
```python
for query in queries:
    search_results = self.serper.search(query=query, num_results=num_results)
    # Deduplicate by URL
    for result in search_results:
        if result.link not in seen_urls:
            all_search_results.append(result)
            seen_urls.add(result.link)
```

#### Step 3: Scrape Top Results
- Proceeds with existing scraping logic
- Uses deduplicated results
- Includes query generation metadata in evidence

## Benefits

### 1. **More Relevant Results**
- Domain-aware queries retrieve more authoritative sources
- Example: For "Deutsche Bank upgraded 3M", searches for "Deutsche Bank 3M rating 2025" will find analyst reports, financial news, and official announcements

### 2. **Better Coverage**
- Multiple optimized queries (up to 3) cover different aspects of the statement
- Deduplication ensures no redundant scraping

### 3. **Improved Authority**
- Targeted queries are more likely to return authoritative sources (e.g., financial institutions, official announcements, major news outlets)
- Better than generic verbatim searches

### 4. **Reduced Noise**
- Focused queries reduce irrelevant results
- Entity extraction ensures searches target verifiable facts

## Usage

### Basic Usage (Automatic)
```python
from src.factchecker.simple.modules.judge_module import JudgeModule

judge = JudgeModule()
result = judge.forward("Deutsche Bank upgraded 3M to buy", web_search_enabled=True)

# If web search is triggered, it will automatically use QueryGenerator
# to generate optimized queries before searching
```

### Testing QueryGenerator Independently
```python
import dspy
from src.factchecker.simple.signatures.query_generator import QueryGenerator

query_gen = dspy.ChainOfThought(QueryGenerator)
result = query_gen(statement="Deutsche Bank upgraded 3M to buy")

print(result.reasoning)  # Explanation of query generation strategy
print(result.queries)    # List of 1-3 optimized queries
```

## Test Files

### 1. `test_query_generator.py`
Tests the QueryGenerator signature independently with various statement types.

### 2. `test_judge_enhancement.py`
Tests the complete JudgeModule integration (already existed, now benefits from QueryGenerator).

## Technical Details

### Query Generation Strategy

The QueryGenerator uses Chain-of-Thought reasoning to:

1. **Identify Key Entities**:
   - Companies (e.g., "Deutsche Bank", "3M")
   - People (e.g., "Elon Musk")
   - Dates (e.g., "2025", "January 2025")
   - Specific claims (e.g., "upgraded to buy", "$44 billion")

2. **Formulate Targeted Queries**:
   - Combine entities with action verbs
   - Add temporal context (current year)
   - Use domain-specific terminology
   - Focus on verifiable facts

3. **Optimize for Authority**:
   - Structure queries to match authoritative source patterns
   - Include specific terms that appear in analyst reports, press releases, etc.

### Deduplication Logic

```python
seen_urls = set()
for result in search_results:
    if result.link not in seen_urls:
        all_search_results.append(result)
        seen_urls.add(result.link)
```

This ensures:
- No duplicate URLs are scraped
- Multiple queries can return overlapping sources without redundancy
- Efficient use of API calls

### Evidence Format

The evidence now includes query generation metadata:

```
=== WEB SEARCH RESULTS ===
Generated 3 optimized queries: Deutsche Bank 3M rating 2025, 3M stock upgrade Deutsche Bank, Deutsche Bank analyst ratings 2025

--- Source 1: Title ---
URL: https://example.com
Snippet: ...
Full Content (truncated):
...
```

## Architecture Integration

```
JudgeModule
    ├── judge (ChainOfThought)
    ├── web_judge (ChainOfThought)
    ├── query_generator (ChainOfThought) [NEW]
    ├── serper (SerperService)
    └── firecrawl (FirecrawlService)

Flow:
1. LLM-only judgment
2. If uncertain → _gather_web_evidence()
   a. QueryGenerator generates 1-3 optimized queries
   b. Execute searches for each query
   c. Deduplicate results by URL
   d. Scrape top N results
3. Re-evaluate with evidence
```

## Performance Considerations

- **API Efficiency**: Deduplication reduces redundant scraping
- **Search Quality**: Optimized queries improve result relevance, potentially requiring fewer searches
- **Cost**: Multiple queries (up to 3) may increase search API calls, but deduplication mitigates redundant scraping

## Future Enhancements

Potential improvements:
1. **Adaptive Query Count**: Adjust number of queries based on statement complexity
2. **Query Validation**: Score and rank generated queries before executing
3. **Source Type Targeting**: Add hints for specific source types (e.g., "site:sec.gov" for financial data)
4. **Multi-Language Support**: Generate queries in multiple languages for international topics
5. **Caching**: Cache query generation results for similar statements

## Conclusion

The QueryGenerator implementation significantly enhances the fact-checking system's web search capabilities by:
- Converting verbatim statements into domain-aware search queries
- Improving the authority and relevance of retrieved evidence
- Maintaining efficiency through intelligent deduplication
- Providing transparent query generation reasoning

This enables more effective fact-checking, especially for domain-specific claims that require authoritative sources.
