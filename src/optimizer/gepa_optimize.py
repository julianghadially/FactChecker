"""GEPA optimizer for FactChecker pipeline.

Optimizes the fact-checker to maximize F1 score of the REFUTED class,
with optional secondary metric for precision of SUPPORTED class.

Usage:
    python -m src.optimizer.gepa_optimize --mlflow --auto light 
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import random

import dspy
import mlflow
from tqdm import tqdm

from src.context_.context import openai_key
from src.factchecker.modules.fact_checker_pipeline import FactCheckerPipeline
from src.evaluation.data_loader import load_dataset, FacToolLabelSchema
from src.evaluation.metrics import calculate_metrics, print_metrics, get_f1, EvaluationMetrics


def load_dspy_examples(path: str, limit: Optional[int] = None) -> list[dspy.Example]:
    """Load dataset as DSPy Examples."""
    dataset = load_dataset(path=path, limit=limit)
    examples = []
    for ex in dataset.examples:
        normalized_label = FacToolLabelSchema.normalize_ground_truth(ex.label)
        examples.append(
            dspy.Example(statement=ex.claim, label=normalized_label).with_inputs("statement")
        )
    return examples


def get_prediction_label(pred) -> str:
    """Extract and normalize prediction label from pipeline output."""
    verdict = pred.overall_verdict if hasattr(pred, 'overall_verdict') else str(pred)
    return FacToolLabelSchema.normalize_prediction(verdict)


def gepa_metric(gold: dspy.Example, pred, trace=None, pred_name=None, pred_trace=None) -> dict:
    """GEPA metric optimizing for REFUTED F1 + SUPPORTED precision.

    Returns score and feedback for GEPA's reflective optimization.
    """
    pred_label = get_prediction_label(pred)
    
    if pred_label == gold.label:
        score = 1.0
        feedback = f"Correct! True label is {gold.label} and predicted label is {pred_label}"
    elif pred_label == "UNKNOWN":
        score = 0.5
        feedback = f"Neutral! It is okay to predict unsupported label when there is no evidence to support or refute the claim. True label is {gold.label} and predicted label is {pred_label}"
    else:
        score = 0
        feedback = f"Incorrect! True label is {gold.label} and predicted label is {pred_label}"
    return dspy.Prediction(score=score, feedback=feedback)

def evaluate_program(program: dspy.Module, examples: list[dspy.Example], name: str) -> EvaluationMetrics:
    """Evaluate program and return metrics using the evaluation module."""
    predictions, ground_truth = [], []

    for ex in tqdm(examples, desc=f"Evaluating {name}"):
        try:
            pred = program(statement=ex.statement)
            predictions.append(pred.overall_verdict if hasattr(pred, 'overall_verdict') else str(pred))
        except Exception as e:
            print(f"Error: {e}")
            predictions.append("ERROR")
        ground_truth.append(ex.label)

    metrics = calculate_metrics(predictions, ground_truth, FacToolLabelSchema)
    print_metrics(metrics, name)
    return metrics


def run_optimization(
    auto: str = "light",
    reflection_model: str = "openai/gpt-5-mini",
    model: str = "openai/gpt-5-mini",
    output_dir: str = "results/optimization",
    num_threads: int = 5, #firecrawl concurrency limit is 5
    use_mlflow: bool = False
):
    """Run GEPA optimization on the FactChecker pipeline."""
    # MLflow setup
    if use_mlflow:
        print("Enabling MLflow tracking...")
        mlflow.dspy.autolog(log_compiles=False, log_evals=True, log_traces_from_compile=True)
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("GEPA-Optimization")

    print("=" * 60)
    print("GEPA Optimization for FactChecker Pipeline")
    print("=" * 60)

    # Configure DSPy
    dspy.configure(lm=dspy.LM(model, api_key=openai_key))

    # Load datasets
    print("\nLoading datasets...")
    all_train = load_dspy_examples("data/FacTool_QA_train.jsonl")
    random.seed(42)
    random.shuffle(all_train)
    split_idx = max(10, len(all_train) // 5)
    valset, trainset = all_train[:split_idx], all_train[split_idx:]
    testset = load_dspy_examples("data/FacTool_QA_test.jsonl")

    print(f"Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")

    # Initialize pipeline
    program = FactCheckerPipeline()

    # Baseline evaluation
    #print("\n" + "=" * 60)
    #print("BASELINE EVALUATION")
    #print("=" * 60)
    #baseline_metrics = evaluate_program(program, testset, "Unoptimized")

    # GEPA optimization
    print("\n" + "=" * 60)
    print(f"GEPA OPTIMIZATION (auto={auto}, reflection={reflection_model}, model={model})")
    print("=" * 60)

    optimizer = dspy.GEPA(
        metric=gepa_metric,
        auto=auto,
        num_threads=num_threads,
        track_stats=True,
        reflection_minibatch_size=3,
        reflection_lm=dspy.LM(model=reflection_model, temperature=1.0, max_tokens=32000, api_key=openai_key)
    )

    optimized_program = optimizer.compile(program, trainset=trainset, valset=valset)

    # Optimized evaluation
    print("\n" + "=" * 60)
    print("OPTIMIZED EVALUATION")
    print("=" * 60)
    optimized_metrics = evaluate_program(optimized_program, testset, "Optimized")

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    optimized_f1 = get_f1(optimized_metrics, "REFUTED")
    print(f"REFUTED F1:         {optimized_f1:.1%}")
    print(f"REFUTED Precision:  {optimized_metrics.per_class_precision.get('REFUTED', 0):.1%}")
    print(f"REFUTED Recall:     {optimized_metrics.per_class_recall.get('REFUTED', 0):.1%}")
    print(f"SUPPORTED Precision:{optimized_metrics.per_class_precision.get('SUPPORTED', 0):.1%}")
    print(f"Accuracy:           {optimized_metrics.accuracy:.1%}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {"auto": auto, "reflection_model": reflection_model},
        "optimized": {"refuted_f1": optimized_f1, "refuted_precision": optimized_metrics.per_class_precision.get('REFUTED', 0), "refuted_recall": optimized_metrics.per_class_recall.get('REFUTED', 0), "supported_precision": optimized_metrics.per_class_precision.get('SUPPORTED', 0), "accuracy": optimized_metrics.accuracy}
    }

    results_file = output_path / f"gepa_optimization_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    program_file = output_path / f"optimized_program_{timestamp}.json"
    optimized_program.save(str(program_file))
    print(f"Program saved to: {program_file}")

    # MLflow logging
    if use_mlflow:
        with mlflow.start_run(run_name=f"GEPA_{auto}_{timestamp}"):
            mlflow.log_params({"auto": auto, "reflection_model": reflection_model})
            mlflow.log_metrics({
                "optimized_refuted_f1": optimized_f1,
                "optimized_refuted_precision": optimized_metrics.per_class_precision.get('REFUTED', 0),
                "optimized_refuted_recall": optimized_metrics.per_class_recall.get('REFUTED', 0),
                "optimized_supported_precision": optimized_metrics.per_class_precision.get('SUPPORTED', 0),
                "optimized_accuracy": optimized_metrics.accuracy
            })
            mlflow.log_artifact(str(results_file))
            mlflow.log_artifact(str(program_file))
        print("MLflow run logged")

    return optimized_program, results


def main():
    parser = argparse.ArgumentParser(description="GEPA Optimizer for FactChecker")
    parser.add_argument("--auto", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--reflection-model", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--model", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--output-dir", type=str, default="results/optimization")
    parser.add_argument("--numthreads", type=int, default=4)
    parser.add_argument("--mlflow", action="store_true")

    args = parser.parse_args()
    run_optimization(
        auto=args.auto,
        reflection_model=args.reflection_model,
        model=args.model,
        output_dir=args.output_dir,
        num_threads=args.numthreads,
        use_mlflow=args.mlflow
    )


if __name__ == "__main__":
    main()
