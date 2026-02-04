# URL Integration Usage Examples

## Quick Start

### Example 1: Evaluate JudgeModule with URL-Enhanced Dataset

```python
from src.optimizer.gepa_optimize import load_dspy_examples, evaluate_program
from src.factchecker.simple.modules.judge_module import JudgeModule
import dspy
from src.context_.context import openai_key

# Configure DSPy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key=openai_key))

# Load dataset with URLs (CSV format)
examples = load_dspy_examples("data/FactChecker_news_claims.csv", limit=10)

# Create JudgeModule (has Firecrawl support)
judge = JudgeModule()

# Evaluate - URLs will be automatically passed and scraped
metrics = evaluate_program(judge, examples, "URL-Enhanced Evaluation")

# Results will show how well the model performs when given reference URLs
print(f"Accuracy: {metrics.accuracy:.2%}")
print(f"REFUTED F1: {metrics.per_class_f1.get('REFUTED', 0):.2%}")
```

### Example 2: Manually Call JudgeModule with URLs

```python
from src.factchecker.simple.modules.judge_module import JudgeModule
import dspy
from src.context_.context import openai_key

# Setup
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key=openai_key))
judge = JudgeModule()

# Fact-check with reference URLs
statement = "Alaska Airlines launched new flights between Seattle and London in May 2026."
urls = [
    "https://www.cbsnews.com/news/joseph-emerson-alaska-airlines-pilot-flight-deck-audio-police-video/",
    "https://nypost.com/2025/12/09/us-news/wild-new-cockpit-audio-reveals-moment-alaska-airlines-pilot-tried-to-crash-plane-mid-flight/"
]

result = judge(statement=statement, urls=urls)

print(f"Verdict: {result.overall_verdict}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")
```

### Example 3: Run GEPA Optimization with URL Dataset

```python
from src.optimizer.gepa_optimize import run_optimization

# Run optimization using URL-enhanced dataset
optimized_program, results = run_optimization(
    auto="light",
    reflection_model="openai/gpt-4o-mini",
    model="openai/gpt-4o-mini",
    output_dir="results/optimization",
    num_threads=5,
    use_mlflow=True
)

# Note: You'll need to modify the dataset paths in run_optimization
# to use FactChecker_news_claims.csv instead of FacTool_QA_train.jsonl
```

### Example 4: Compare Performance With and Without URLs

```python
from src.optimizer.gepa_optimize import load_dspy_examples, evaluate_program
from src.factchecker.simple.modules.judge_module import JudgeModule
import dspy
from src.context_.context import openai_key

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key=openai_key))

# Load examples with URLs
examples_with_urls = load_dspy_examples("data/FactChecker_news_claims.csv", limit=20)

# Create two separate judge instances
judge = JudgeModule()

# Evaluate WITH URLs (using modified examples)
print("Evaluating WITH URLs...")
metrics_with_urls = evaluate_program(judge, examples_with_urls, "With URLs")

# Evaluate WITHOUT URLs (strip URLs from examples)
print("\nEvaluating WITHOUT URLs...")
examples_no_urls = []
for ex in examples_with_urls:
    # Create new example without URLs
    examples_no_urls.append(
        dspy.Example(statement=ex.statement, label=ex.label).with_inputs("statement")
    )
metrics_no_urls = evaluate_program(judge, examples_no_urls, "Without URLs")

# Compare results
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"Accuracy with URLs:    {metrics_with_urls.accuracy:.2%}")
print(f"Accuracy without URLs: {metrics_no_urls.accuracy:.2%}")
print(f"Improvement:           {(metrics_with_urls.accuracy - metrics_no_urls.accuracy):.2%}")
```

### Example 5: Create Custom Dataset with URLs

```python
import json

# Create JSONL dataset with URLs
dataset = [
    {
        "claim": "The Eiffel Tower is located in Paris, France.",
        "label": "true",
        "urls": [
            "https://en.wikipedia.org/wiki/Eiffel_Tower",
            "https://www.toureiffel.paris/en"
        ]
    },
    {
        "claim": "The Great Wall of China is visible from space.",
        "label": "false",
        "url": "https://www.nasa.gov/vision/space/workinginspace/great_wall.html"
    }
]

# Save to JSONL
with open("data/custom_dataset.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")

# Use it
from src.optimizer.gepa_optimize import load_dspy_examples
examples = load_dspy_examples("data/custom_dataset.jsonl")

print(f"Loaded {len(examples)} examples")
for ex in examples:
    print(f"  Statement: {ex.statement}")
    if hasattr(ex, 'urls'):
        print(f"  URLs: {len(ex.urls)}")
```

### Example 6: Create Custom CSV Dataset with URLs

```python
import csv

# Create CSV dataset
data = [
    {
        "claim": "Bitcoin was created by Satoshi Nakamoto.",
        "label": "TRUE",
        "url": "https://bitcoin.org/bitcoin.pdf, https://en.wikipedia.org/wiki/Bitcoin"
    },
    {
        "claim": "The Moon is made of cheese.",
        "label": "FALSE",
        "url": "https://www.nasa.gov/moon"
    }
]

# Save to CSV
with open("data/custom_dataset.csv", "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["claim", "label", "url"])
    writer.writeheader()
    writer.writerows(data)

# Use it
from src.optimizer.gepa_optimize import load_dspy_examples
examples = load_dspy_examples("data/custom_dataset.csv")

print(f"Loaded {len(examples)} examples")
for ex in examples:
    print(f"  Statement: {ex.statement}")
    if hasattr(ex, 'urls'):
        print(f"  URLs: {ex.urls}")
```

## Advanced Usage

### Custom Evaluation Loop with URL Handling

```python
from src.factchecker.simple.modules.judge_module import JudgeModule
from src.optimizer.gepa_optimize import load_dspy_examples
import dspy
from src.context_.context import openai_key
from tqdm import tqdm

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", api_key=openai_key))

# Load examples
examples = load_dspy_examples("data/FactChecker_news_claims.csv", limit=50)

# Initialize judge
judge = JudgeModule()

# Custom evaluation with detailed logging
results = []
for ex in tqdm(examples, desc="Evaluating"):
    try:
        # Check if example has URLs
        if hasattr(ex, 'urls') and ex.urls:
            print(f"\nProcessing with {len(ex.urls)} URLs...")
            pred = judge(statement=ex.statement, urls=ex.urls)
        else:
            pred = judge(statement=ex.statement)

        results.append({
            'statement': ex.statement,
            'true_label': ex.label,
            'predicted': pred.overall_verdict,
            'confidence': pred.confidence,
            'reasoning': pred.reasoning,
            'had_urls': hasattr(ex, 'urls') and bool(ex.urls)
        })
    except Exception as e:
        print(f"Error: {e}")
        results.append({
            'statement': ex.statement,
            'true_label': ex.label,
            'predicted': 'ERROR',
            'had_urls': hasattr(ex, 'urls') and bool(ex.urls)
        })

# Analyze results
correct = sum(1 for r in results if r['predicted'] == r['true_label'])
print(f"\nAccuracy: {correct/len(results):.2%}")

# Compare URL vs non-URL performance
url_results = [r for r in results if r.get('had_urls')]
no_url_results = [r for r in results if not r.get('had_urls')]

if url_results:
    url_correct = sum(1 for r in url_results if r['predicted'] == r['true_label'])
    print(f"Accuracy with URLs: {url_correct/len(url_results):.2%}")

if no_url_results:
    no_url_correct = sum(1 for r in no_url_results if r['predicted'] == r['true_label'])
    print(f"Accuracy without URLs: {no_url_correct/len(no_url_results):.2%}")
```

## Dataset Format Requirements

### JSONL Format (both supported)

```json
{"claim": "...", "label": "true", "url": "https://example.com/1, https://example.com/2"}
{"claim": "...", "label": "false", "urls": ["https://example.com/1", "https://example.com/2"]}
```

### CSV Format

```csv
claim,label,url
"Statement text",TRUE,"https://example.com/1, https://example.com/2"
"Another statement",FALSE,"https://example.com/3"
```

## Tips

1. **Rate Limiting**: Firecrawl has rate limits (default 5 concurrent). Use `num_threads=5` or lower.

2. **URL Quality**: Ensure URLs are valid and accessible. Failed scrapes are handled gracefully.

3. **Cost Considerations**: Each URL scrape costs API credits. Test with small datasets first.

4. **Timeout Handling**: Long-running evaluations should use tqdm progress bars to monitor progress.

5. **Caching**: Consider implementing URL caching if evaluating multiple times on same dataset.

## Troubleshooting

### URLs Not Being Passed
```python
# Check if example has URLs
ex = examples[0]
print(f"Has urls attribute: {hasattr(ex, 'urls')}")
print(f"URLs: {ex.urls if hasattr(ex, 'urls') else 'None'}")
```

### Firecrawl Errors
```python
# Test Firecrawl directly
from src.services.firecrawl_service import FirecrawlService
service = FirecrawlService()
result = service.scrape("https://example.com")
print(f"Success: {result.success}")
print(f"Error: {result.error}")
```

### Dataset Format Issues
```python
# Check raw data
import json
with open("data/your_dataset.jsonl", "r") as f:
    first_line = json.loads(f.readline())
    print(f"Keys: {first_line.keys()}")
    print(f"Has URL field: {'url' in first_line or 'urls' in first_line}")
```
