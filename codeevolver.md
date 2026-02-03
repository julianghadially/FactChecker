# Architecture Summary: FactChecker with JudgeModule Optimization

## High-Level Purpose
FactChecker is a DSPy-based fact verification system that assesses factual correctness of claims using external evidence through iterative web search. The system is being optimized using **GEPA (Generalized Evolutionary Prompt-based Agent)** to maximize the **F1 score of the REFUTED class** with secondary optimization for SUPPORTED class precision. The entry point `JudgeModule` represents a simplified, barebones fact-checker that evaluates statements directly using LLM knowledge without web research.

## Key Modules and Responsibilities

### Entry Point: JudgeModule (Simple Path)
- **Location**: `src.factchecker.simple.modules.judge_module.JudgeModule`
- **Purpose**: Fast, barebones fact-checker without external research
- **Components**: Uses DSPy ChainOfThought with Judge signature
- **Output**: Direct verdict (SUPPORTED | CONTAINS_UNSUPPORTED_CLAIMS | CONTAINS_REFUTED_CLAIMS) with confidence and reasoning

### Full FactCheckerPipeline (Research Path)
1. **ClaimExtractorModule**: Decomposes statements into atomic factual claims
2. **FireJudgeModule**: Iteratively evaluates each claim (max 3 iterations), requesting research when needed
3. **ResearchAgentModule**: Orchestrates evidence gathering via:
   - Serper web search
   - LLM-guided PageSelector (selects up to 3 relevant pages)
   - Firecrawl scraping
   - EvidenceSummarizer (extracts relevant facts)
4. **AggregatorModule**: Combines claim-level verdicts using priority logic:
   - Any refuted → CONTAINS_REFUTED_CLAIMS
   - Any unsupported → CONTAINS_UNSUPPORTED_CLAIMS
   - All supported → SUPPORTED

### Evaluation & Optimization
- **Metrics Module**: Calculates accuracy, precision/recall per class, confusion matrices, F1 scores
- **GEPA Optimizer**: Reflective optimization framework that:
  - Uses trainset/valset splits from FacTool_QA dataset
  - Provides scored feedback (1.0 correct, 0.5 neutral for UNKNOWN, 0.0 incorrect)
  - Optimizes prompts through reflection with configurable intensity (light/medium/heavy)
  - Tracks per-class performance, especially REFUTED F1 and SUPPORTED precision

## Data Flow

**Training/Optimization Flow**:
```
FacTool_QA_train.jsonl → Load DSPy Examples → Split (trainset/valset)
                                                        ↓
                                        GEPA Optimizer (gepa_metric feedback)
                                                        ↓
                                        JudgeModule/FactCheckerPipeline
                                                        ↓
                                        Optimized Program → Evaluate on testset
```

**Inference Flow (Simple)**:
```
Statement → JudgeModule → ChainOfThought(Judge) → {verdict, confidence, reasoning}
```

**Inference Flow (Full)**:
```
Statement → ClaimExtractor → Claims[] → FireJudge (iterative) ⇄ ResearchAgent
                                                                    ↓
                                                    Aggregator → Final Verdict
```

## Metric Being Optimized

**Primary**: `gepa_metric` function optimizes for:
- **Exact match accuracy** with graded feedback
- Score=1.0 for correct predictions
- Score=0.5 for UNKNOWN predictions (acceptable when evidence insufficient)
- Score=0.0 for incorrect predictions

**Target Performance Metrics** (from evaluations):
- REFUTED F1 score
- REFUTED precision & recall
- SUPPORTED precision
- Overall accuracy on predictions (excluding UNKNOWN)

The metric normalizes predictions using `FacToolLabelSchema` to handle label variations and provides feedback for GEPA's reflective learning process, enabling iterative prompt optimization that improved performance from 91% to 96% accuracy on news claims.
