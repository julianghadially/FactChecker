# FactChecker Architecture Summary

## High-Level Purpose
This is a **fact-checking system** built with DSPy that evaluates the factual accuracy of statements. The entry point `JudgeModule` is a simplified version that performs direct LLM-based fact verification without external research, serving as a lightweight alternative to the full research-enabled pipeline.

## Key Modules and Responsibilities

### 1. **JudgeModule (Entry Point - Simple Version)**
- **Location**: `src/factchecker/simple/modules/judge_module.py`
- **Purpose**: Barebones fact checker that judges statements using only LLM knowledge
- **Components**: Uses `dspy.ChainOfThought` with `Judge` signature
- **Output**: Returns verdict (SUPPORTED/CONTAINS_UNSUPPORTED_CLAIMS/CONTAINS_REFUTED_CLAIMS), confidence score, and reasoning

### 2. **FactCheckerPipeline (Full Version)**
- **Location**: `src/factchecker/modules/fact_checker_pipeline.py`
- **Purpose**: Complete research-enabled fact-checking with iterative web search
- **Orchestrates**: Claim extraction → Iterative research → Verdict aggregation
- **Submodules**:
  - `ClaimExtractorModule`: Breaks statements into individual factual claims
  - `FireJudgeModule`: FIRE (Fact-checking with Iterative Research and Evaluation) - evaluates claims with adaptive web search
  - `ResearchAgentModule`: Executes web searches, selects pages intelligently, scrapes content, and extracts relevant evidence
  - `AggregatorModule`: Combines claim-level verdicts into overall statement verdict

### 3. **ResearchAgentModule**
- **Purpose**: Web-based evidence gathering with LLM-guided page selection
- **Integration**: Uses SerperService (search) and FirecrawlService (scraping)
- **Intelligence**: LLM selects which pages to visit and extracts claim-relevant evidence

## Data Flow

```
Statement Input
    ↓
[Simple Path: JudgeModule]
    → LLM evaluates directly using internal knowledge
    → Returns verdict + confidence + reasoning

[Full Path: FactCheckerPipeline]
    → ClaimExtractor splits into claims
    → For each claim:
        → FireJudge iterates (max 3 iterations):
            → Evaluates current evidence
            → If insufficient → generates search query
            → ResearchAgent executes search
                → LLM selects relevant pages (max 3 visits)
                → Scrapes and extracts evidence
            → Returns verdict when sufficient evidence found
    → Aggregator combines claim verdicts
    → Returns overall verdict + confidence + reasoning
```

## Metric Being Optimized

**Metric**: `gepa_metric` (defined in `src/optimizer/gepa_optimize.py`)

**Optimization Goal**: Maximize **F1 score for REFUTED class** + improve SUPPORTED precision

**Scoring Logic**:
- **Correct prediction** (pred == gold): score = 1.0
- **UNKNOWN prediction** (hedging): score = 0.5 (neutral, acceptable when evidence insufficient)
- **Incorrect prediction**: score = 0.0

**Optimizer**: Uses DSPy's GEPA (Gradual Evolution with Performance Assessment) to optimize prompts and module behavior through reflective learning on train/validation sets, evaluated on FacTool QA dataset.

The architecture emphasizes **accuracy-cost tradeoffs**: JudgeModule for fast/cheap evaluation vs. FactCheckerPipeline for research-backed verification with adaptive iteration and intelligent page selection to minimize API costs.
