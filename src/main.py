"""Main entry point for the FactChecker system."""

import argparse
import dspy

from context.context import openai_key, serper_key, firecrawl_key
from factchecker.modules.fact_checker_pipeline import FactCheckerPipeline
from baseline.baseline_model import BaselineModel
from evaluation.evaluate import run_evaluation


def configure_dspy(model: str = "openai/gpt-4o-mini"):
    """Configure DSPy with the specified model.

    Args:
        model: Model identifier (e.g., 'openai/gpt-4o-mini', 'anthropic/claude-3-sonnet').
    """
    lm = dspy.LM(model, api_key=openai_key)
    dspy.configure(lm=lm)


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


def run_benchmark(sample_size: int, model: str):
    """Run benchmark evaluation on HOVER dataset.

    Args:
        sample_size: Number of examples to evaluate.
        model: Model to use for evaluation.
    """
    configure_dspy(model)

    fact_checker = FactCheckerPipeline(
        serper_api_key=serper_key,
        firecrawl_api_key=firecrawl_key
    )
    baseline = BaselineModel()

    run_evaluation(
        fact_checker=fact_checker,
        baseline_model=baseline,
        sample_size=sample_size
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
        default="openai/gpt-4o-mini",
        help="Model to use (e.g., 'openai/gpt-4o-mini', 'anthropic/claude-3-sonnet')"
    )

    args = parser.parse_args()

    if args.mode == "check":
        if not args.statement:
            parser.error("--statement is required for 'check' mode")
        run_single_check(args.statement, args.model)

    elif args.mode == "evaluate":
        run_benchmark(args.sample_size, args.model)


if __name__ == "__main__":
    main()
