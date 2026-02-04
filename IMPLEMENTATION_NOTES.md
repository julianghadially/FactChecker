# Implementation Notes: Multi-Query Search Enhancement

## Design Decisions

### 1. Query Count: 1-3 Queries
**Rationale:**
- **Minimum 1:** Ensures at least one search is performed (fallback to original statement)
- **Maximum 3:** Balances thoroughness with API cost and latency
  - More queries = higher Serper API costs
  - More queries = longer response time
  - 3 queries typically cover: main claim + temporal aspect + entity details

**Trade-off Analysis:**
```
1 query  → Fast, cheap, may miss evidence
3 queries → Moderate cost/speed, good coverage
5+ queries → Expensive, slow, diminishing returns
```

### 2. Source Count: 3-4 Total Sources
**Rationale:**
- Increased from 2 to 3-4 for better coverage
- Still within LLM context limits (4 × 5000 chars = 20K chars max)
- Provides multiple perspectives without overwhelming the Judge
- Deduplication ensures diverse sources

**Context Window Considerations:**
```
Current: 4 sources × 5000 chars = 20,000 chars
DSPy typical context: 100K+ tokens (Claude, GPT-4)
Remaining for reasoning: 80K+ tokens
✓ Well within limits
```

### 3. URL Deduplication Strategy
**Implementation:**
```python
seen_urls = set()
if result.link not in seen_urls:
    all_results.append(result)
    seen_urls.add(result.link)
```

**Why URL-based:**
- Simple and effective
- Prevents duplicate scraping costs
- Different queries may return same authoritative source
- Ensures diverse information sources

**Alternative considered but rejected:**
- Domain deduplication: Too restrictive (e.g., multiple good Wikipedia articles)
- Content similarity: Too expensive (requires embedding comparison)
- Title deduplication: Unreliable (same content, different titles)

### 4. Query Generation Approach
**Using ChainOfThought:**
```python
self.query_generator = dspy.ChainOfThought(QueryGenerator)
```

**Why ChainOfThought:**
- Provides reasoning about query selection
- Better quality queries than direct prediction
- Can explain why certain aspects were targeted
- Minimal overhead (one LLM call)

**Alternative considered:**
- Predict: Faster but lower quality
- ReAct: Too complex for this task
- ProgramOfThought: Overkill for query generation

### 5. Search Results Per Query: 3 Results
**Rationale:**
```python
search_results = self.serper.search(query=query, num_results=3)
```

- 3 queries × 3 results = 9 potential sources
- After deduplication: typically 3-4 unique sources
- Good balance between coverage and API costs
- Top 3 results usually contain most relevant information

**Cost Analysis:**
```
Before: 1 query × 2 results = 2 API calls (Serper + Firecrawl)
After:  3 queries × 3 results = 9 Serper calls, ~4 Firecrawl calls
Cost increase: ~2-3x (acceptable for accuracy improvement)
```

## Error Handling & Fallbacks

### 1. Query Generation Failure
```python
if not queries:
    queries = [statement]  # Fallback to original behavior
```

**Scenarios:**
- LLM returns empty list
- QueryGenerator module fails
- Network timeout

**Response:** Use original statement as single query

### 2. Search Failure
```python
except Exception as e:
    print(f"Warning: Evidence gathering failed: {e}")
    return ""
```

**Scenarios:**
- Serper API down
- Network issues
- Rate limiting

**Response:** Return empty evidence, trigger fallback in `forward()` method

### 3. Scraping Failure
```python
if scraped.success and scraped.markdown:
    evidence_parts.append(...)
```

**Scenarios:**
- Firecrawl API down
- URL unreachable
- Paywall or blocked content

**Response:** Skip failed source, continue with remaining sources

### 4. No Results Found
```python
if not all_results:
    return ""
```

**Response:** Return empty evidence, Judge uses first pass result with note

## Performance Considerations

### Latency Analysis

**Before Enhancement:**
```
First pass:     ~1s  (Judge LLM call)
Search:         ~1s  (1 Serper query)
Scrape:         ~3s  (2 Firecrawl calls)
Second pass:    ~1s  (Judge LLM call with evidence)
─────────────────────
Total:          ~6s
```

**After Enhancement:**
```
First pass:     ~1s  (Judge LLM call)
Query gen:      ~1s  (QueryGenerator LLM call)
Search:         ~2s  (3 Serper queries, could be parallel)
Scrape:         ~5s  (4 Firecrawl calls, could be parallel)
Second pass:    ~1s  (Judge LLM call with evidence)
─────────────────────
Total:          ~10s (sequential)
With parallel:  ~7s  (if search/scrape parallelized)
```

**Mitigation Strategies:**
1. Keep max sources at 4 (don't increase further)
2. Consider parallelizing search calls (future enhancement)
3. Consider parallelizing Firecrawl calls (future enhancement)
4. Cache frequently accessed URLs (future enhancement)

### Cost Analysis

**API Costs Per Fact Check:**

Before:
- 2 × LLM calls (Judge): ~$0.002
- 1 × Serper call: ~$0.001
- 2 × Firecrawl calls: ~$0.002
- **Total: ~$0.005**

After:
- 3 × LLM calls (Judge + QueryGen): ~$0.003
- 3 × Serper calls: ~$0.003
- 4 × Firecrawl calls: ~$0.004
- **Total: ~$0.010**

**Cost increase: 2x** (acceptable for improved accuracy)

## Testing Strategy

### Unit Tests
```python
def test_query_generator():
    # Test query generation with various statements
    # - Temporal claims
    # - Numeric claims
    # - Entity-focused claims
    # - Mixed claims

def test_deduplication():
    # Test URL deduplication logic
    # - Duplicate URLs from different queries
    # - All unique URLs
    # - Mix of duplicates and uniques

def test_fallback_behavior():
    # Test fallback when query generation fails
    # Test fallback when search fails
    # Test behavior with zero results
```

### Integration Tests
```python
def test_end_to_end():
    # Test full flow with real statement
    # Verify query generation → search → scrape → judge

def test_verdict_improvement():
    # Compare verdicts before and after enhancement
    # Focus on statements that should be REFUTED
```

### Regression Tests
```python
def test_backward_compatibility():
    # Ensure existing functionality unchanged
    # Test trigger conditions still work
    # Test output format unchanged
```

## Configuration Options (Future)

Consider making these configurable:

```python
class JudgeModuleConfig:
    max_queries: int = 3              # Max queries to generate
    max_sources: int = 4              # Max sources to scrape
    results_per_query: int = 3        # Results per search query
    max_source_length: int = 5000     # Max chars per source
    enable_query_generation: bool = True  # Feature flag
    enable_deduplication: bool = True     # Feature flag
```

**Benefits:**
- A/B testing different configurations
- Cost optimization for different use cases
- Easy rollback if issues arise
- Gradual rollout strategy

## Monitoring & Observability

### Metrics to Track

1. **Query Generation Metrics:**
   - Average number of queries generated
   - Query generation success rate
   - Query generation latency

2. **Search Metrics:**
   - Total search API calls per fact check
   - Search result coverage
   - Deduplication rate (duplicates found / total results)

3. **Quality Metrics:**
   - Verdict distribution (SUPPORTED/UNSUPPORTED/REFUTED)
   - Confidence score changes (first pass vs second pass)
   - Evidence retrieval success rate

4. **Performance Metrics:**
   - End-to-end latency
   - Component latency breakdown
   - API failure rates

### Logging Strategy

```python
# Add structured logging (future enhancement)
logger.info("Query generation", {
    "statement": statement[:100],
    "queries_generated": len(queries),
    "queries": queries
})

logger.info("Search execution", {
    "total_queries": len(queries),
    "total_results": len(all_results),
    "duplicates_removed": duplicates_count,
    "sources_to_scrape": len(all_results[:4])
})

logger.info("Evidence gathering complete", {
    "sources_scraped": len(evidence_parts),
    "total_evidence_length": len(evidence)
})
```

## Future Enhancements

### Short-term (Low Effort, High Value)
1. **Parallel search execution:** Execute all queries concurrently
2. **Parallel scraping:** Scrape all URLs concurrently
3. **Query generation caching:** Cache queries for similar statements

### Medium-term (Moderate Effort)
1. **Query quality scoring:** Rank and select best queries
2. **Source diversity scoring:** Prioritize diverse domains
3. **Evidence relevance filtering:** Filter out irrelevant sources before scraping
4. **Configuration system:** Make parameters configurable

### Long-term (High Effort)
1. **Adaptive query generation:** Adjust query count based on statement complexity
2. **Query result analysis:** Analyze search snippets before scraping
3. **Iterative refinement:** Generate follow-up queries based on initial results
4. **Evidence summarization:** Summarize evidence before passing to Judge

## Migration Path

### Phase 1: Deployment (Current)
- ✅ Implement multi-query enhancement
- ✅ Maintain backward compatibility
- ✅ Add basic error handling

### Phase 2: Monitoring (Next)
- Add logging and metrics
- Monitor verdict distribution
- Track latency and costs
- Collect user feedback

### Phase 3: Optimization (Future)
- Implement parallel execution
- Add configuration system
- Fine-tune parameters based on metrics
- Consider query caching

### Phase 4: Advanced Features (Long-term)
- Adaptive query generation
- Evidence quality scoring
- Iterative refinement
- Custom query strategies per claim type

## Rollback Plan

If issues arise:

1. **Quick Disable:**
   ```python
   # Add feature flag in __init__
   self.enable_multi_query = os.getenv('ENABLE_MULTI_QUERY', 'true').lower() == 'true'

   # In _gather_evidence
   if not self.enable_multi_query:
       # Use old single-query logic
       search_results = self.serper.search(query=statement, num_results=2)
       ...
   ```

2. **Gradual Rollback:**
   - Reduce max_queries to 1
   - Reduce max_sources to 2
   - Effectively reverts to original behavior

3. **Full Rollback:**
   - Revert changes to judge_module.py
   - Remove query_generator imports
   - QueryGenerator module remains but unused

## Success Criteria

### Primary Metrics:
- ✅ REFUTED verdicts increase by 20%+ for factually incorrect statements
- ✅ CONTAINS_UNSUPPORTED_CLAIMS verdicts decrease by 15%+ when evidence exists
- ✅ Average confidence scores increase by 0.1+ for research-triggered cases

### Secondary Metrics:
- ✅ End-to-end latency remains under 15 seconds
- ✅ Evidence retrieval success rate > 80%
- ✅ API failure rate < 5%

### Quality Checks:
- ✅ No regression in existing test cases
- ✅ Generated queries are specific and relevant
- ✅ Deduplication working correctly (no duplicate URLs)
