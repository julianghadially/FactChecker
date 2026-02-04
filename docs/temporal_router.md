# Temporal Router Module

## Overview

The `TemporalRouterModule` is an intelligent routing layer that sits between user input and the fact-checking system. It analyzes statements to determine whether they require web research (via `FactCheckerPipeline`) or can be handled by the fast `JudgeModule` using only LLM knowledge.

## Architecture

```
User Input → TemporalRouterModule → Decision
                                   ↓
                    ┌──────────────┴──────────────┐
                    ↓                             ↓
              JudgeModule                  FactCheckerPipeline
           (Fast evaluation)              (Web research enabled)
```

## Key Features

### 1. Temporal Analysis
The router analyzes statements for temporal references to determine if they fall within or beyond the LLM's knowledge cutoff (June 2024):

- **Date Extraction**: Parses various date formats:
  - `YYYY-MM-DD` and `YYYY/MM/DD` (e.g., "2025-01-15")
  - `Month DD, YYYY` (e.g., "January 15, 2025")
  - `Mon DD, YYYY` (e.g., "Jan 15, 2025")
  - `DD Month YYYY` (e.g., "15 January 2025")
  - Year references (e.g., "in 2025", "year 2024")

- **Temporal Keywords**: Detects keywords suggesting recent/current events:
  - Time references: `today`, `yesterday`, `tomorrow`, `now`, `present`, `current`
  - Relative time: `this week/month/year`, `last week/month/year`, `next week/month/year`
  - Recency indicators: `recent`, `recently`, `latest`, `upcoming`
  - Post-cutoff years: `2024`, `2025`, `2026`

### 2. URL Detection
Automatically extracts and processes URLs from statements, routing to web research when URLs are present.

### 3. Priority URL Support
Accepts an optional list of priority URLs that should be scraped first before performing web searches, allowing users to provide specific evidence sources.

## Routing Logic

The router uses a rule-based decision system:

### Route to FactCheckerPipeline (Web Research) if:
1. **URLs are provided** (either in statement or as explicit parameter)
2. **Dates beyond knowledge cutoff** are detected (≥ June 2024)
3. **Temporal keywords** suggesting recent/current events are present

### Route to JudgeModule (Fast) if:
1. **No URLs** present
2. **All dates** (if any) are before knowledge cutoff
3. **No temporal keywords** suggesting recent events

## Usage

### Basic Usage

```python
from src.factchecker.modules.temporal_router_module import TemporalRouterModule

# Initialize router
router = TemporalRouterModule()

# Fact-check a statement
result = router(statement="The Apollo 11 mission landed on the moon in 1969.")

# Check routing decision
print(f"Route: {result.route_decision}")  # "judge" or "pipeline"
print(f"Reason: {result.route_reason}")
print(f"Verdict: {result.overall_verdict}")
```

### With Priority URLs

```python
# Provide specific evidence URLs
urls = [
    "https://example.com/earnings-report",
    "https://example.com/news-article"
]

result = router(
    statement="Company X reported record profits in Q4 2024.",
    urls=urls
)
```

### Configuration

```python
# Customize router parameters
router = TemporalRouterModule(
    max_judge_iterations=3,      # Max search iterations in pipeline
    max_page_visits=3,           # Max pages to visit per search
    knowledge_cutoff=datetime(2024, 6, 1)  # Custom cutoff date
)
```

## Return Value

The router returns a `dspy.Prediction` object with:

### Common Fields (both routes):
- `statement`: The input statement
- `overall_verdict`: `SUPPORTED` | `CONTAINS_UNSUPPORTED_CLAIMS` | `CONTAINS_REFUTED_CLAIMS`
- `confidence`: Float between 0.0 and 1.0
- `reasoning`: Explanation of the verdict
- `route_decision`: `"judge"` or `"pipeline"`
- `route_reason`: Why that route was chosen

### Additional Fields (pipeline route only):
- `claims`: List of extracted claims
- `claim_results`: Detailed results for each claim with evidence and search queries

## Examples

### Example 1: Historical Fact (Routes to Judge)

```python
statement = "The Apollo 11 mission landed on the moon on July 20, 1969."
result = router(statement=statement)

# Output:
# Route: judge
# Reason: No temporal references or URLs requiring web research
# Verdict: SUPPORTED
```

### Example 2: Recent Event (Routes to Pipeline)

```python
statement = "In January 2025, global tech companies announced major layoffs."
result = router(statement=statement)

# Output:
# Route: pipeline
# Reason: Date beyond knowledge cutoff: 2025-01-01 >= 2024-06-01
# Verdict: (depends on web research)
```

### Example 3: Current Events (Routes to Pipeline)

```python
statement = "The latest climate report shows record temperatures this year."
result = router(statement=statement)

# Output:
# Route: pipeline
# Reason: Temporal keywords suggest recent/current events
# Verdict: (depends on web research)
```

### Example 4: With URLs (Routes to Pipeline)

```python
statement = "According to the report, unemployment rates have decreased."
urls = ["https://example.com/employment-report-2024"]
result = router(statement=statement, urls=urls)

# Output:
# Route: pipeline
# Reason: URLs provided (1 URLs found)
# Verdict: (depends on web research from provided URL)
```

## Integration with Research Agent

The `ResearchAgentModule` has been enhanced to support priority URLs:

```python
# Priority URLs are scraped first
result = research_agent(
    claim="Company profits increased",
    query="company quarterly earnings",
    priority_urls=["https://example.com/earnings"]
)
```

### Priority URL Processing

1. **Scrape priority URLs first** (up to `max_page_visits` limit)
2. **Extract evidence** from each priority URL
3. **Continue to web search** if:
   - Page visits budget remains
   - Strong evidence (supports/refutes) not yet found
4. **Skip web search** if priority URLs exhaust the page visit budget

## Performance Considerations

### Cost Optimization
- **JudgeModule route**: Single LLM call (fast, cheap)
- **Pipeline route**: Multiple LLM calls + API calls (slower, more expensive)
- The router minimizes unnecessary web research for historical facts

### Latency
- **JudgeModule**: ~1-3 seconds
- **Pipeline**: ~10-30 seconds (depends on search iterations)

### Accuracy Trade-off
- **JudgeModule**: Relies on LLM knowledge only (may be outdated for recent events)
- **Pipeline**: Uses current web data (more accurate for recent events)

## Testing

Run the demo script to see the router in action:

```bash
python examples/temporal_router_demo.py
```

The demo includes:
1. Date extraction testing (no API calls required)
2. Routing decision examples for various statement types
3. Priority URL demonstration

## Configuration in main.py

The main entry point has been updated to use `TemporalRouterModule`:

```python
# Old
from src.factchecker.modules.fact_checker_pipeline import FactCheckerPipeline
fact_checker = FactCheckerPipeline()

# New
from src.factchecker.modules.temporal_router_module import TemporalRouterModule
fact_checker = TemporalRouterModule()
```

## Implementation Details

### Date Parsing

The router uses regex patterns to extract dates from various formats:

```python
# Supported formats
"2025-01-15"           # ISO format
"January 15, 2025"     # Month-first
"15 January 2025"      # Day-first
"in 2025"              # Year only
```

### URL Extraction

URLs are extracted using a comprehensive regex pattern:

```python
url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
```

### Temporal Keyword Matching

Case-insensitive regex matching for temporal indicators:

```python
keywords = [
    r'\btoday\b',
    r'\brecent(ly)?\b',
    r'\blatest\b',
    r'\b2024\b',
    # ... etc
]
```

## Limitations

1. **Date Format Coverage**: May not catch all date formats (e.g., non-English months)
2. **False Positives**: Some temporal keywords might appear in historical contexts
3. **Knowledge Cutoff**: Assumes June 2024 cutoff (adjust for different models)
4. **URL Quality**: Doesn't validate if provided URLs are accessible or relevant

## Future Enhancements

Potential improvements:

1. **ML-based Routing**: Use a classifier instead of rules
2. **Confidence Scoring**: Return routing confidence scores
3. **Hybrid Mode**: Combine both approaches for high-stakes claims
4. **Feedback Loop**: Learn from routing mistakes
5. **Multi-language Support**: Detect dates/keywords in other languages
6. **URL Validation**: Pre-check if URLs are accessible before routing

## API Reference

### TemporalRouterModule

```python
class TemporalRouterModule(dspy.Module):
    def __init__(
        self,
        max_judge_iterations: int = 3,
        max_page_visits: int = 3,
        knowledge_cutoff: Optional[datetime] = None
    ):
        """Initialize the temporal router.

        Args:
            max_judge_iterations: Max search iterations per claim in pipeline.
            max_page_visits: Max pages to visit per search query in pipeline.
            knowledge_cutoff: Custom knowledge cutoff date (defaults to June 2024).
        """

    def forward(
        self,
        statement: str,
        urls: Optional[list[str]] = None
    ) -> dspy.Prediction:
        """Route the fact-checking request to appropriate module.

        Args:
            statement: The statement to fact-check.
            urls: Optional list of URLs to use as priority evidence sources.

        Returns:
            dspy.Prediction with verdict, confidence, reasoning, and routing info.
        """
```

### Helper Methods

```python
def _extract_urls(self, text: str) -> list[str]:
    """Extract URLs from text."""

def _extract_dates(self, text: str) -> list[datetime]:
    """Extract and parse dates from text."""

def _has_temporal_keywords(self, text: str) -> bool:
    """Check if text contains temporal keywords."""

def _should_use_web_research(
    self,
    statement: str,
    urls: list[str],
    dates: list[datetime]
) -> tuple[bool, str]:
    """Determine if web research is needed."""
```

## See Also

- [JudgeModule Documentation](./judge_module.md)
- [FactCheckerPipeline Documentation](./fact_checker_pipeline.md)
- [ResearchAgentModule Documentation](./research_agent_module.md)
