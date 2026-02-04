# Fact-Checking Bot Requirements
FactChecker is a project that implements DSPy in Python to build an Agent as a judge fact checker. This agent as a judge is used to assess the factual correctness of the output of another language model. 

Key Problem: Language models hallucinate, and LLM as a judge hallucination detectors share a lot of the same biases as the main model. Therefore, LLM as a Judge is not a reliable factual correctness judges. 

Solution: This agent as a judge is tasked with assessing factual correctness (including detecting hallucinations). The key strategy is to factually ground the hallucination detector with external data. The agent is armed with multi-hop web search to assess the validity of claims.

## Objective
Build a compound AI system using DSPy that performs multi-hop fact verification against the HOVER dataset, comparing performance against a simple baseline.

## Must-Haves
- **Two systems**: 
  1. Agent-as-a-judge fact-checker (DSPy system)
  2. Baseline model (single LLM query)
- **Dataset**: HOVER dataset for evaluation
- **Model provider**: gpt-5-mini
- **Metrics**: Accuracy, Precision of positive label, precision of negative label
- **Output**: Comparison report showing FactChecker performance against hover label

## Factchecker compound AI system architecture
- **Input** Accepts language model statements
- **Output** Classifies claims as supported or not_supported or refuted

## Requirements for Searching and Fetching Web Results
If you choose to search and fetch web pages
- Web search and page fetching system should be somewhat cost conscious. 
- A language model should intelligently filter out unnecessary page visits
- The scaffolding should allow us to reap efficiency gains in the future
- Implementation of page fetching can either be via MCP server tool or internal function. 

## Architecture Constraints
- Use DSPy signatures for fact-checking logic
- Multi-hop retrieval for evidence gathering is allowed
- Evaluation script that runs both systems on same data

## Success Criteria
- Runs end-to-end on HOVER dev set
- Generates comparison metrics
- Code is maintainable and well-documented
- Can easily swap model providers

## Out of Scope
- Production deployment
- Real-time API
- Custom retrievers (use existing)