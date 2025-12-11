# FactChecker

FactChecker is a DSPy-based multi-hop fact verification system that assesses the factual correctness of language model outputs. Unlike simple LLM-as-judge approaches that share biases with the models they evaluate, FactChecker grounds its judgments in external evidence through iterative web search.

**Results Summary:**
1. DSPy + GEPA optimization improved FactChecker performance relative to the unoptimized model. On current event claims, FactChecker went from 91% to 96% accuracy on predictions made, and added 10-18 percentage points to each class-specific recall
2. FactChecker verifies and refutes way more claims than the baseline model. On current event claims, FactChecker tied the base model on accuracy on predictions made, but added 21-66 percentage points to each class-specific recall
3. On the 2023 FacTools-QA dataset, All systems performed roughly the same accuracy on predictions cases: 89% vs 88% vs. 91% for optimized, unoptimized, and baseline, respectively. It is likely that GPT-5-mini has been trained on this data set, rendering the test unhelpful. 

## Results

### FactChecker News Claims

#### FactChecker, Optimized

|                                    | Supported | Refuted |
|------------------------------------|-----------|---------|
| **'Supported' Prediction**         | 23        | 1       |
| **No Prediction ('Unsupported')**  | 15        | 5       |
| **'Refuted' Predictions**          | 1         | 30      |

**Accuracy on predicted cases: 96% (55 preditions)** ⭐️
**Supported recall: 59%** ⭐️
**Refuted recall: 83%** ⭐️
Model: GPT-5-Mini

#### Baseline - FactChecker, Not Optimized

|                                    | Supported | Refuted |
|------------------------------------|-----------|---------|
| **'Supported' Prediction**         | 17        | 4       |
| **No Prediction ('Unsupported')**  | 18        | 10      |
| **'Refuted' Predictions**          | 0         | 26      |

**Accuracy on predicted cases: 91% (47 preditions)** 
**Supported recall: 49%** 
**Refuted recall: 65%** 
Model: GPT-5-Mini

#### Baseline - GPT-5-Mini

|                                    | Supported | Refuted |
|------------------------------------|-----------|---------|
| **'Supported' Prediction**         | 15        | 0       |
| **No Prediction ('Unsupported')**  | 24        | 27      |
| **'Refuted' Predictions**          | 1         | 8       |

**Accuracy on predicted cases: 96% (24 preditions)** 
**Supported recall: 38%** 
**Refuted recall: 23%** 
Model: GPT-5-Mini




### FacTools
This assessment uses GPT five mini for every component of the system, including the reflection model, FactChecker model, and baseline model. See "notes" in FacTools-QA for seven claims that were flipped due to no longer being true. 

#### FactChecker (Optimized)

|                            | Supported | Refuted |
|----------------------------|-----------|---------|
| **Correct Predictions**    | 68        | 5       |
| **Incorrect Predictions**  | 5         | 17      |

**Accuracy on Predicted Cases: 89%** 
Model: GPT-5-Mini

#### Baseline

|                            | Supported | Refuted |
|----------------------------|-----------|---------|
| **Correct Predictions**    | 77        | 2       |
| **Incorrect Predictions**  | 7         | 14      |

**Accuracy on Predicted Cases: 91%** 
Model: GPT-5-Mini





## Architecture

```
Statement → [ClaimExtractor] → Claims[]
                                  │
                            For each claim:
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │      FireJudge         │
                     │  (Iterative Evaluator) │
                     └───────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
              Has verdict?              Need more info?
                    │                         │
                    │                         ▼
                    │            ┌────────────────────────┐
                    │            │    ResearchAgent       │
                    │            │  ┌──────────────────┐  │
                    │            │  │ Serper Search    │  │
                    │            │  └────────┬─────────┘  │
                    │            │           ▼            │
                    │            │  ┌──────────────────┐  │
                    │            │  │ PageSelector     │  │
                    │            │  │ (LLM-guided)     │  │
                    │            │  └────────┬─────────┘  │
                    │            │           ▼            │
                    │            │  ┌──────────────────┐  │
                    │            │  │ Firecrawl Scrape │  │
                    │            │  └────────┬─────────┘  │
                    │            │           ▼            │
                    │            │  ┌──────────────────┐  │
                    │            │  │EvidenceSummarizer│  │
                    │            │  └──────────────────┘  │
                    │            └────────────┬───────────┘
                    │                         │
                    │              evidence ──┘
                    │                         │
                    └─────────┬───────────────┘
                              │
                              ▼
                     ┌────────────────────────┐
                     │      Aggregator        │
                     │  Priority Logic:       │
                     │  1. Any refuted →      │
                     │     CONTAINS_REFUTED   │
                     │  2. Any unsupported →  │
                     │     CONTAINS_UNSUPPORTED│
                     │  3. All supported →    │
                     │     SUPPORTED          │
                     └───────────┬────────────┘
                                 │
                                 ▼
                          Final Verdict
```

## Components

| Component | Type | Description |
|-----------|------|-------------|
| **ClaimExtractor** | Signature/Module | Extracts individual factual claims from a statement |
| **FireJudge** | Signature/Module | Iteratively evaluates claims, requesting searches or returning verdicts |
| **ResearchAgent** | Module | Orchestrates web search, page selection, scraping, and evidence extraction |
| **PageSelector** | Signature | LLM-guided selection of which pages to visit (max 3 per query) |
| **EvidenceSummarizer** | Signature | Extracts relevant facts from scraped pages |
| **Aggregator** | Signature/Module | Combines claim verdicts into overall statement verdict |





## Installation

```bash
pip install dspy-ai requests firecrawl-py tqdm
```

Set environment variables:
```bash
export OPENAI_AGENTJUDGEJG_KEY="your-openai-key"
export SERPER_KEY="your-serper-key"
export FIRECRAWL_KEY="your-firecrawl-key"
```

## Usage

### Single Statement Check
```bash
python src/main.py --mode check --statement "The Eiffel Tower was built in 1889"
```

### Run HOVER Benchmark
```bash
python src/main.py --mode evaluate --sample-size 100
```

### Use Different Model
```bash
python src/main.py --mode evaluate --model anthropic/claude-3-sonnet
```

## Project Structure

```
src/
├── services/                  # External API integrations
│   ├── serper_service.py      # Google search via Serper
│   └── firecrawl_service.py   # Page scraping via Firecrawl
├── factchecker/
│   ├── signatures/            # DSPy signatures (input/output specs)
│   │   ├── claim_extractor.py
│   │   ├── fire_judge.py
│   │   ├── page_selector.py
│   │   ├── evidence_summarizer.py
│   │   └── aggregator.py
│   ├── modules/               # DSPy modules (execution logic)
│   │   ├── claim_extractor_module.py
│   │   ├── fire_judge_module.py
│   │   ├── research_agent_module.py
│   │   ├── aggregator_module.py
│   │   └── fact_checker_pipeline.py
│   └── models/                # Data types
│       └── data_types.py
├── baseline/                  # Simple LLM baseline for comparison
│   └── baseline_model.py
├── evaluation/                # Benchmarking system
│   ├── data_loader.py         # HOVER dataset loader
│   ├── metrics.py             # Accuracy, precision calculations
│   └── evaluate.py            # Comparison script
└── main.py                    # CLI entry point
```

## Key Resources
- **DSPy docs**: https://dspy.ai/
- **HOVER paper**: 
- **FIRE LLM Fact-Checking paper**: FIRE stands for Fact-checking with Iterative Retrieval and Verification: https://aclanthology.org/2025.findings-naacl.158.pdf?utm_source=chatgpt.com
- **FIRE paper github**: https://github.com/mbzuai-nlp/fire
- **LoCal LLM Fact-checking Paper**: https://dl.acm.org/doi/10.1145/3696410.3714748
 

