# CodeEvolver Details

## Overview

CodeEvolver offers autonomous coding agents for high reliability AI systems. It uses GEPA optimization to evolve your AI system code until it performs optimally for a given dataset and outcome metric. See specs/CodeEvolver_analysis.md

## Documentation


### evaluate.py specs

```python
#!/usr/bin/env python3
"""Evaluation script for CodeEvolver GEPA optimization.

CONTRACT:
    python eval/evaluate.py \
        --candidate /tmp/candidate.json \
        --batch /tmp/batch.json \
        --output /tmp/results.json

INPUT FILES (written by CodeEvolver):
    candidate.json: {"predictor.predict": "instruction text", ...}
    batch.json: [{"statement": "...", "label": "..."}, ...]

OUTPUT FILE (written by your script):
    results.json: {"scores": [1.0, 0.0, ...], "outputs": [...]}
"""

import argparse
import json

# =============================================================================
# YOUR IMPORTS - Add your project imports here
# =============================================================================

# from src.your_module import YourPipeline
# from src.your_metrics import compute_accuracy


# =============================================================================
# IMPLEMENT THESE FUNCTIONS
# =============================================================================

def load_program(candidate: dict):
    """Load your program and apply candidate prompt configurations.
    
    Args:
        candidate: {"predictor.predict": "instruction text", ...}
    
    Returns:
        Your initialized program ready to run.
    """
    # TODO: Load your DSPy pipeline
    # TODO: Apply candidate prompts to predictors
    raise NotImplementedError("Implement load_program()")


def run_and_score(program, example: dict) -> tuple[dict, float]:
    """Run program on one example and compute score.
    
    Args:
        program: Your initialized program
        example: Single example from batch (e.g., {"statement": "...", "label": "..."})
    
    Returns:
        (output_dict, score) where score is 0.0-1.0
    """
    # TODO: Run your program on the example
    # TODO: Compute score against ground truth
    raise NotImplementedError("Implement run_and_score()")


# =============================================================================
# MAIN - No changes needed below
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    with open(args.candidate) as f:
        candidate = json.load(f)
    with open(args.batch) as f:
        batch = json.load(f)
    
    program = load_program(candidate)
    
    scores, outputs = [], []
    for example in batch:
        try:
            output, score = run_and_score(program, example)
            outputs.append(output)
            scores.append(score)
        except Exception as e:
            outputs.append({"error": str(e)})
            scores.append(0.0)
    
    with open(args.output, "w") as f:
        json.dump({"scores": scores, "outputs": outputs}, f)


if __name__ == "__main__":
    main()
```



### run_job.py specs

```python
#!/usr/bin/env python3
"""Create a CodeEvolver GEPA optimization job.

USAGE:
    python scripts/create_job.py

REQUIRED FILES IN YOUR REPO:
    - eval/evaluate.py  (evaluation script, see template)
    - data/train.json   (training dataset)

ENVIRONMENT:
    CODEEVOLVER_API_KEY - Your API key (or pass directly)
"""

import os
import requests

# =============================================================================
# CONFIGURATION - Customize these for your project
# =============================================================================

CODEEVOLVER_API_URL = "https://codeevolver.modal.run"

JOB_CONFIG = {
    # Required
    "repo_url": "https://github.com/YOUR_ORG/YOUR_REPO",
    "trainset_path": "data/train.json",
    "eval_script": "eval/evaluate.py",
    
    # Optional
    "valset_path": "data/val.json",  # or None
    "config": {
        "max_iterations": 100,
        "reflection_lm": "openai/gpt-5-mini",
    }
}

# =============================================================================
# JOB CREATION
# =============================================================================

def create_job(api_key: str = None) -> dict:
    """Create optimization job via CodeEvolver API."""
    api_key = api_key or os.environ.get("CODEEVOLVER_API_KEY")
    if not api_key:
        raise ValueError("CODEEVOLVER_API_KEY not set")
    
    response = requests.post(
        f"{CODEEVOLVER_API_URL}/optimize",
        json=JOB_CONFIG,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_job_status(job_id: str, api_key: str = None) -> dict:
    """Check job status."""
    api_key = api_key or os.environ.get("CODEEVOLVER_API_KEY")
    
    response = requests.get(
        f"{CODEEVOLVER_API_URL}/job/{job_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    job = create_job()
    print(f"Job created: {job['job_id']}")
    print(f"Status: {job['status']}")
```