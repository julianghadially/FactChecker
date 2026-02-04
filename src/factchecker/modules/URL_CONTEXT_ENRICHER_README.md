# URL Context Enricher Module

## Overview

The `UrlContextEnricherModule` is a preprocessing module that wraps `JudgeModule` and automatically enriches statements with evidence from provided URLs before evaluation. This enables the judge to make evidence-based verdicts instead of relying solely on the LLM's internal knowledge.

## Key Features

- **Automatic URL Scraping**: Uses FirecrawlService to scrape web pages
- **Context Enrichment**: Prepends extracted facts to the statement before evaluation
- **Flexible Input**: Accepts single URL or multiple URLs
- **Non-invasive**: Returns results with the original statement (not the enriched version)
- **Error Handling**: Gracefully handles scraping failures and continues with other URLs
- **Configurable**: Customizable URL limit and character extraction limit

## Architecture

```
Statement + URLs → URL Scraping → Context Enrichment → JudgeModule → Verdict
                 (FirecrawlService)                   (unchanged interface)
```

The module sits between the input and JudgeModule, transparently enriching the context without modifying the judge's signature or interface.

## Usage

### Basic Usage

```python
from src.factchecker.modules import UrlContextEnricherModule

# Initialize with default settings
enricher = UrlContextEnricherModule()

# Evaluate a statement with URL evidence
statement = "Python 3.12 was released in October 2023."
url = "https://www.python.org/downloads/release/python-3120/"

result = enricher.forward(statement, url=url)

print(f"Verdict: {result.overall_verdict}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")
```

### With Multiple URLs

```python
# Provide multiple URLs for broader context
statement = "The James Webb Space Telescope launched in December 2021."
urls = [
    "https://en.wikipedia.org/wiki/James_Webb_Space_Telescope",
    "https://www.nasa.gov/mission_pages/webb/main/index.html"
]

result = enricher.forward(statement, urls=urls)
```

### Custom Configuration

```python
from src.factchecker.simple.modules.judge_module import JudgeModule

# Create with custom settings
custom_judge = JudgeModule()  # Optional: use a pre-configured judge
enricher = UrlContextEnricherModule(
    judge=custom_judge,
    max_urls=3,  # Process up to 3 URLs
    max_chars_per_url=1500  # Extract up to 1500 chars per URL
)
```

### Without URLs (Falls Back to Standard Judge)

```python
# When no URLs provided, behaves like standard JudgeModule
statement = "The Earth orbits the Sun."
result = enricher.forward(statement)  # Uses LLM knowledge only
```

## API Reference

### `UrlContextEnricherModule`

#### `__init__(judge=None, max_urls=2, max_chars_per_url=1000)`

Initialize the URL context enricher module.

**Parameters:**
- `judge` (JudgeModule, optional): JudgeModule instance to wrap. Creates new one if None.
- `max_urls` (int): Maximum number of URLs to scrape. Default: 2.
- `max_chars_per_url` (int): Maximum characters to extract per URL. Default: 1000.

#### `forward(statement, url=None, urls=None)`

Evaluate a statement with optional URL context enrichment.

**Parameters:**
- `statement` (str): The statement to evaluate.
- `url` (str, optional): Single URL to scrape for context.
- `urls` (list[str], optional): List of URLs to scrape for context.

**Returns:**
- `dspy.Prediction` with:
  - `statement` (str): The original input statement
  - `overall_verdict` (str): SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS
  - `confidence` (float): Confidence score between 0.0 and 1.0
  - `reasoning` (str): Explanation of the verdict

## How It Works

1. **URL Collection**: Collects URLs from `url` and/or `urls` parameters
2. **URL Scraping**: Uses FirecrawlService to scrape the first `max_urls` URLs
3. **Fact Extraction**: Extracts up to `max_chars_per_url` characters from each page
4. **Context Building**: Formats scraped content as:
   ```
   Context from provided sources:
   [URL1]: [extracted facts]
   [URL2]: [extracted facts]

   Statement to evaluate: {original_statement}
   ```
5. **Judge Evaluation**: Passes enriched statement to JudgeModule
6. **Result Mapping**: Returns result with original statement (not enriched version)

## Error Handling

The module gracefully handles various error scenarios:

- **Scraping Failures**: If a URL fails to scrape, an error message is included in the context and processing continues with other URLs
- **Network Issues**: Caught and logged, doesn't stop evaluation
- **Invalid URLs**: Handled by FirecrawlService's URL cleaning utilities
- **No URLs Provided**: Falls back to standard JudgeModule behavior

## Benefits Over Standard JudgeModule

1. **Evidence-Based Verdicts**: Uses actual web content instead of just LLM knowledge
2. **Reduced "UNSUPPORTED" Verdicts**: Can verify claims that are outside the LLM's training data
3. **Transparent**: Same interface as JudgeModule - drop-in replacement
4. **Flexible**: Can work with or without URLs
5. **Cost-Effective**: Only scrapes when URLs are provided

## Integration with Existing Pipeline

The module can be used as a drop-in replacement for JudgeModule in existing code:

```python
# Before
from src.factchecker.simple.modules import JudgeModule
judge = JudgeModule()
result = judge.forward(statement)

# After - with URL support
from src.factchecker.modules import UrlContextEnricherModule
judge = UrlContextEnricherModule()
result = judge.forward(statement, url=url)  # Now supports URLs!
```

## Performance Considerations

- **Scraping Time**: Each URL scrape takes ~1-3 seconds (dependent on Firecrawl API)
- **Token Cost**: Enriched context increases input tokens to LLM
- **Recommended Settings**:
  - `max_urls=2`: Balance between evidence breadth and cost
  - `max_chars_per_url=1000`: Provides sufficient context without token explosion

## Future Enhancements

Potential improvements for future versions:

1. **LLM-Based Summarization**: Use LLM to extract key facts instead of truncation
2. **Relevance Filtering**: Only include content relevant to the statement
3. **Parallel Scraping**: Scrape multiple URLs concurrently
4. **Caching**: Cache scraped content to avoid redundant API calls
5. **Source Attribution**: Include which URL supported/refuted the verdict

## Examples

See `examples/url_context_enricher_example.py` for complete working examples.

## Dependencies

- `dspy`: Core framework
- `src.factchecker.simple.modules.judge_module.JudgeModule`: Underlying judge
- `src.services.firecrawl_service.FirecrawlService`: Web scraping service
