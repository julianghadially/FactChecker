# FactChecker Architecture Summary

## High-Level Purpose
FactChecker is a DSPy-based fact verification system that assesses the factual correctness of statements by grounding judgments in external evidence through iterative web search. Unlike simple LLM-as-judge approaches that share biases with the models they evaluate, FactChecker performs web research to verify or refute claims.

The entry point `JudgeModule` is a simplified version that evaluates statements directly using LLM knowledge without external research—serving as a faster alternative when evidence gathering is not required.

## Key Modules and Responsibilities

### Full Pipeline (FactCheckerPipeline)
1. **ClaimExtractorModule**: Decomposes input statements into individual factual claims
2. **FireJudgeModule**: Iteratively evaluates each claim using the FIRE (Fact-checking with Iterative Retrieval and Evaluation) approach—requests web searches until reaching a verdict or exhausting the search budget (max 3 iterations)
3. **ResearchAgentModule**: Orchestrates evidence gathering through:
   - Serper web search (fetches top 10 results)
   - PageSelector (LLM-guided selection of most relevant pages, max 3 per query)
   - Firecrawl scraping (extracts page content as markdown)
   - EvidenceSummarizer (extracts relevant facts with stance: supports/refutes/neutral)
4. **AggregatorModule**: Combines claim-level verdicts into overall statement verdict using priority logic (any refuted → CONTAINS_REFUTED_CLAIMS; any unsupported → CONTAINS_UNSUPPORTED_CLAIMS; all supported → SUPPORTED)

### Simplified Entry Point (JudgeModule)
- Direct statement evaluation without claim extraction or research
- Uses ChainOfThought reasoning to output verdict, confidence score (0.0-1.0), and reasoning

## Data Flow
```
Statement → ClaimExtractor → Claims[] → For each claim:
  → FireJudge (iterative loop):
    → Needs evidence? → ResearchAgent → Serper Search → PageSelector → Firecrawl Scrape → EvidenceSummarizer → Evidence
    → Has verdict? → Return verdict
  → AggregatorModule → Final Verdict (SUPPORTED/CONTAINS_UNSUPPORTED_CLAIMS/CONTAINS_REFUTED_CLAIMS)
```

## Metric Being Optimized
The system uses **GEPA (Generative Prompt Augmentation)** optimization targeting:
- **Primary**: F1 score of the REFUTED class
- **Secondary**: Precision of the SUPPORTED class
- **Scoring**: Correct predictions = 1.0, UNKNOWN predictions = 0.5 (neutral), Incorrect = 0.0

The metric function (`gepa_metric` in `src/optimizer/gepa_optimize.py`) provides reflective feedback to optimize DSPy prompts, improving from 91% to 96% accuracy and adding 10-18 percentage points to class-specific recall on current event claims.
