#!/usr/bin/env python3
"""Job creation script for CodeEvolver GEPA optimization.

Submits an optimization job to the CodeEvolver API and polls for results.

SETUP:
    1. Update JOB_CONFIG below with your repository details
    2. Set CODEEVOLVER_API_KEY environment variable
    3. Run: python -m src.optimizer.codeevolver_job

WHAT YOU NEED IN YOUR REPO:
    - A DSPy module class (entry_point)
    - A metric function (metric) that scores predictions
    - A training dataset (trainset_path)
    - A program.json from dspy program.save() (optional)

CodeEvolver handles the rest: loading your module, applying prompt
mutations, running evaluation, and optimizing via GEPA.

The FactChecker pipeline has the following optimizable components:
    - claim_extractor.extractor: Extract factual claims from statements
    - fire_judge.judge: Evaluate claims with FIRE methodology
    - research_agent.page_selector: Select relevant pages to scrape
    - research_agent.evidence_summarizer: Summarize evidence from web pages
    - aggregator.aggregator: Aggregate claim verdicts into overall verdict
"""

import os
import sys
import json
import time
import requests

# =============================================================================
# CONFIGURATION - FactChecker specific settings
# =============================================================================

CODEEVOLVER_API_URL = "https://julianghadially--codeevolver-fastapi-app-dev.modal.run"
REPO_URL = "https://github.com/julianghadially/FactChecker"

JOB_CONFIG = {
    # Required - Repository
    "repo_url": REPO_URL,

    # Required - DSPy module class (dotted import path)
    "program": "src.factchecker.modules.fact_checker_pipeline.FactCheckerPipeline",  # Changed from entry_point

    # Required - Metric function (dotted import path)
    "metric": "src.codeevolver.metric.metric",

    # Required - Training data (path to file in your repo)
    "trainset_path": "data/FacTool_QA_train.jsonl",

    # Optional - Validation set (defaults to trainset if not provided)
    # "valset_path": "data/FacTool_QA_test.jsonl",

    # Optional - Saved DSPy program state
    # "saved_program_json_path": "results/optimization/optimized_program.json",  # Changed from program_json_path

    # Optional - Field names that are inputs (vs. labels)
    # For FacTool dataset: 'claim' is input, 'label' is ground truth
    "input_keys": ["claim"],

    # Optimization configuration
    "reflection_lm": "openai/gpt-5-mini",  # Removed task_lm (not in schema)
    "max_metric_calls": 1000,
    "num_threads": 5,  # Limited by firecrawl concurrency
    "seed": 42,
}

# =============================================================================
# JOB CREATION AND MANAGEMENT
# =============================================================================

def create_job(api_key: str = None, config_override: dict = None) -> dict:
    """Create optimization job via CodeEvolver API.

    Args:
        api_key: CodeEvolver API key (or set CODEEVOLVER_API_KEY env var)
        config_override: Optional dict to override JOB_CONFIG values

    Returns:
        API response with job_id, status, etc.
    """
    api_key = api_key or os.environ.get("CODEEVOLVER_API_KEY")
    if not api_key:
        print("Error: CODEEVOLVER_API_KEY not set")
        sys.exit(1)

    # Merge config with overrides
    config = {**JOB_CONFIG}
    if config_override:
        config.update(config_override)

    print(f"Creating optimization job...")
    print(f"  Repository: {config['repo_url']}")
    print(f"  Entry point: {config['entry_point']}")
    print(f"  Metric: {config['metric']}")
    print(f"  Train data: {config.get('trainset_path', 'inline')}")
    print(f"  Max metric calls: {config.get('max_metric_calls', 1000)}")

    response = requests.post(
        f"{CODEEVOLVER_API_URL}/optimize",
        json=config,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def poll_job(job_id: str, api_key: str = None, interval: int = 30) -> dict:
    """Poll job status until completion.

    Args:
        job_id: Job ID from create_job response
        api_key: CodeEvolver API key
        interval: Seconds between polls

    Returns:
        Final job status dict
    """
    api_key = api_key or os.environ.get("CODEEVOLVER_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    print(f"\nPolling job {job_id}...")
    while True:
        response = requests.get(
            f"{CODEEVOLVER_API_URL}/job/{job_id}",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        status = response.json()

        state = status["status"]
        iteration = status.get("current_iteration", "?")
        best_score = status.get("best_score")

        score_str = f"  Best score: {best_score:.4f}" if best_score is not None else ""
        print(f"  [{state}] Iteration {iteration}{score_str}")

        if state in ("completed", "failed", "cancelled"):
            return status

        time.sleep(interval)


def get_job_status(job_id: str, api_key: str = None) -> dict:
    """Get current job status (single check, no polling).

    Args:
        job_id: Job ID to check
        api_key: CodeEvolver API key

    Returns:
        Job status dict
    """
    api_key = api_key or os.environ.get("CODEEVOLVER_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    response = requests.get(
        f"{CODEEVOLVER_API_URL}/job/{job_id}",
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    result = create_job()
    print(f"\nJob created: {result['job_id']}")

    final = poll_job(result["job_id"])
    print(f"\nFinal status: {final['status']}")

    if final["status"] == "completed":
        print(f"Best score: {final.get('best_score')}")
        print(f"Best candidate: {json.dumps(final.get('best_candidate'), indent=2)}")
    elif final.get("error"):
        print(f"Error: {final['error']}")
