"""Main entry point for the FactChecker system."""

import argparse
import dspy
import mlflow

from src.context_.context import openai_key, serper_key, firecrawl_key
from src.factchecker.modules.fact_checker_pipeline import FactCheckerPipeline
from src.baseline.baseline_model import BaselineModel
from src.evaluation.evaluate import run_evaluation


def configure_dspy(model: str = "openai/gpt-5-mini"):
    """Configure DSPy with the specified model.

    Args:
        model: Model identifier (e.g., 'openai/gpt-4o-mini', 'anthropic/claude-3-sonnet').
    """
    lm = dspy.LM(model, api_key=openai_key)
    dspy.configure(lm=lm)
    dspy.configure_cache(
        enable_disk_cache=False,
        enable_memory_cache=False
    )


def run_single_check(statement: str, model: str):
    """Run fact-checking on a single statement.

    Args:
        statement: The statement to fact-check.
        model: Model to use for evaluation.
    """
    configure_dspy(model)

    pipeline = FactCheckerPipeline(
        serper_api_key=serper_key,
        firecrawl_api_key=firecrawl_key
    )

    print(f"\nFact-checking statement: {statement}\n")
    print("-" * 60)

    result = pipeline(statement=statement)

    print(f"\nClaims extracted: {len(result.claims)}")
    for i, claim in enumerate(result.claims, 1):
        cr = result.claim_results[i - 1]
        print(f"\n  {i}. {claim}")
        print(f"     Verdict: {cr.verdict}")
        print(f"     Searches performed: {len(cr.search_queries)}")
        if cr.search_queries:
            for q in cr.search_queries:
                print(f"       - {q}")

    print("\n" + "=" * 60)
    print(f"OVERALL VERDICT: {result.overall_verdict}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Reasoning: {result.reasoning}")
    print("=" * 60)


def run_benchmark(sample_size: int, model: str, optimized_program_path: str = None, dataset_path: str = None):
    """Run benchmark evaluation on dataset.

    Args:
        sample_size: Number of examples to evaluate.
        model: Model to use for evaluation.
        optimized_program_path: Path to optimized program JSON file.
        dataset_path: Path to dataset file (JSON, JSONL, or CSV).
    """
    configure_dspy(model)

    fact_checker = FactCheckerPipeline()
    # Load optimized program if path provided
    if optimized_program_path:
        print(f"Loading optimized program from: {optimized_program_path}")
        fact_checker.load(optimized_program_path)
        print("Optimized program loaded successfully!")
    else:
        print("Using unoptimized program")
    baseline = BaselineModel()

    run_evaluation(
        fact_checker=fact_checker,
        baseline_model=baseline,
        sample_size=sample_size,
        dataset_path=dataset_path,
        num_threads=40  # firecrawl concurrency limit is 50
    )


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="FactChecker - DSPy-based Multi-hop Fact Verification System"
    )
    parser.add_argument(
        "--mode",
        choices=["check", "evaluate"],
        required=True,
        help="Mode: 'check' for single statement, 'evaluate' for HOVER benchmark"
    )
    parser.add_argument(
        "--statement",
        type=str,
        help="Statement to fact-check (required for 'check' mode)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of examples for evaluation (default: 100)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-5-mini",
        help="Model to use (e.g., 'openai/gpt-5-mini', 'anthropic/claude-4.5-sonnet')"
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow tracking (default: False)"
    )
    parser.add_argument(
        "--optimized-program-path",
        type=str,
        default="results/optimization/optimized_program_20251208_045645.json",
        help="Path to optimized program JSON (e.g., results/optimization/optimized_program_20241208.json)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/FactChecker_news_claims.csv",
        help="Path to dataset file (JSON, JSONL, or CSV). Default: data/FactChecker_news_claims.csv"
    )

    args = parser.parse_args()

    if args.mlflow:
        print("Enabling MLflow tracking")
        mlflow.dspy.autolog(
            log_compiles=True,
            log_evals=True,
            log_traces_from_compile=True
        )
        # Configure MLflow tracking
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("DSPy-Optimization")
    
    optimized_program_path = args.optimized_program_path
    if str(optimized_program_path).lower() in ["none", "null", ""]:
        optimized_program_path = None

    if args.mode == "check":
        if not args.statement:
            parser.error("--statement is required for 'check' mode")
        run_single_check(args.statement, args.model)

    elif args.mode == "evaluate":
        run_benchmark(args.sample_size, args.model, optimized_program_path, args.dataset_path)

if __name__ == "__main__":
    main()
