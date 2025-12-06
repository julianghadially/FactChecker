# Fact-Checking Bot Requirements
FactChecker is a project that implements DSPy in Python to build an Agent as a judge fact checker. This agent as a judge is used to assess the factual correctness of the output of another language model. 

Key Problem: Language models hallucinate, and LLM as a judge hallucination detectors share a lot of the same biases as the main model. Therefore, LLM as a Judge is not a reliable factual correctness judges. 

Solution: This agent as a judge is tasked with assessing factual correctness (including detecting hallucinations). The key strategy is to factually ground the hallucination detector with external data. The agent is armed with multi-hop web search to assess the validity of claims.

## Objective
Build a compound AI system using DSPy that performs multi-hop fact verification against the HOVER dataset, comparing performance against a simple baseline.

## Must-Haves
- **Two systems**: 
  1. Agent-as-a-judge fact-checker (DSPy with multi-hop reasoning)
  2. Baseline model (single LLM query)
- **Dataset**: HOVER dataset for evaluation
- **Model provider**: gpt-5-mini
- **Metrics**: Accuracy, Precision of positive label, precision of negative label
- **Output**: Comparison report showing FactChecker performance against hover label

## Factchecker compound AI system
- **Input** Accepts language model statements
- **Output** Classifies claims as supported or not_supported or refuted
- **Nodes**
    - **Claim Extractor Module:** Extracts a list of claims from a statement 
    - **FIRE Judge:** Iterative fact judge that classifies a claim or generates a search query 
    - **Research Agent** Iteratively picks a page from the search results to visit, and continues to visit once evidence supports or refutes the claim. limited to 3 page visits per search query. Produces a summary 
    - **Aggregator** Take the factual correctness output for each claim, and determines if the overall statement is supported, "contains unsupported claims," or "contains refuted claims." (Note: 1 refuted claim instantly causes the statement to say: "contains refuted claims, even if unsupported claims exist)

## Requirements for Searching and Fetching Web Results
- We require a Web search and page fetching system that is somewhat cost conscious. 
- A language model should intelligently filter out unnecessary page visits
- The scaffolding should allow us to reap efficiency gains in the future
- Implementation of page fetching can either be via MCP server tool or internal function. 

## Architecture Constraints
- Use DSPy signatures for fact-checking logic
- Multi-hop retrieval for evidence gathering
- Modular design: separate retrieval, reasoning, evaluation
- Type hints and docstrings required
- Evaluation script that runs both systems on same data

## Success Criteria
- Runs end-to-end on HOVER dev set
- Generates comparison metrics
- Code is maintainable and well-documented
- Can easily swap model providers

## Out of Scope (for v1)
- Production deployment
- Real-time API
- Custom retrievers (use existing)