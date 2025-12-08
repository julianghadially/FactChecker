"""Evaluation script comparing FactChecker vs Baseline on HOVER dataset."""

import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from .data_loader import load_dataset
from .metrics import calculate_metrics, print_metrics


def run_evaluation(
    fact_checker,
    baseline_model,
    sample_size: int = 100,
    output_dir: str = "results",
    dataset_path: str = "data/FacTool_QA.jsonl"
) -> dict:
    """Run evaluation comparing FactChecker vs Baseline.

    Args:
        fact_checker: Initialized FactCheckerPipeline instance.
        baseline_model: Initialized BaselineModel instance.
        sample_size: Number of examples to evaluate.
        output_dir: Directory to save results.
        dataset_path: Path to dataset file (JSON or JSONL format).

    Returns:
        Dict with evaluation results for both systems.
    """
    # Load dataset (returns DatasetWithSchema)
    dataset_with_schema = load_dataset(path=dataset_path, limit=sample_size)
    dataset = dataset_with_schema.examples
    schema = dataset_with_schema.schema

    print(f"\nUsing label schema: {schema.__name__}")
    print(f"Valid labels: {schema.get_labels()}")

    # Collect predictions
    fc_predictions = []
    baseline_predictions = []
    ground_truth = []
    detailed_results = []

    for example in tqdm(dataset, desc="Evaluating"):
        ground_truth.append(example.label)

        # FactChecker prediction
        try:
            fc_result = fact_checker(statement=example.claim)
            fc_predictions.append(fc_result.overall_verdict)
        except Exception as e:
            fc_predictions.append("ERROR")
            print(f"FactChecker error on {example.uid}: {e}")

        # Baseline prediction
        try:
            bl_result = baseline_model(claim=example.claim)
            baseline_predictions.append(bl_result["verdict"])
        except Exception as e:
            baseline_predictions.append("ERROR")
            print(f"Baseline error on {example.uid}: {e}")

        detailed_results.append({
            "uid": example.uid,
            "claim": example.claim,
            "ground_truth": example.label,
            "factchecker_prediction": fc_predictions[-1],
            "factchecker_normalized": schema.normalize_prediction(fc_predictions[-1]),
            "baseline_prediction": baseline_predictions[-1],
            "baseline_normalized": schema.normalize_prediction(baseline_predictions[-1]),
            "num_hops": example.num_hops
        })

    # Calculate metrics using schema-aware normalization
    fc_metrics = calculate_metrics(fc_predictions, ground_truth, schema)
    bl_metrics = calculate_metrics(baseline_predictions, ground_truth, schema)

    # Prepare results
    results = {
        "timestamp": datetime.now().isoformat(),
        "sample_size": sample_size,
        "label_schema": schema.__name__,
        "labels": schema.get_labels(),
        "factchecker": {
            "accuracy": fc_metrics.accuracy,
            "total_examples": fc_metrics.total_examples,
            "valid_examples": fc_metrics.valid_examples,
            "error_count": fc_metrics.error_count,
            "per_class_precision": fc_metrics.per_class_precision,
            "per_class_recall": fc_metrics.per_class_recall,
            "confusion_matrix": fc_metrics.confusion_matrix
        },
        "baseline": {
            "accuracy": bl_metrics.accuracy,
            "total_examples": bl_metrics.total_examples,
            "valid_examples": bl_metrics.valid_examples,
            "error_count": bl_metrics.error_count,
            "per_class_precision": bl_metrics.per_class_precision,
            "per_class_recall": bl_metrics.per_class_recall,
            "confusion_matrix": bl_metrics.confusion_matrix
        },
        "detailed_results": detailed_results
    }

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"evaluation_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print_metrics(fc_metrics, "FactChecker (Agent-as-Judge with Web Search)")
    print_metrics(bl_metrics, "Baseline (Single LLM Query)")

    improvement = fc_metrics.accuracy - bl_metrics.accuracy
    print(f"\nAccuracy Improvement: {improvement:+.1%}")
    
    # Print key metrics comparison
    print(f"\n{'='*60}")
    print("KEY METRICS COMPARISON")
    print(f"{'='*60}")
    
    # Precision of SUPPORTED
    if "SUPPORTED" in fc_metrics.per_class_precision and "SUPPORTED" in bl_metrics.per_class_precision:
        fc_sup_prec = fc_metrics.per_class_precision["SUPPORTED"]
        bl_sup_prec = bl_metrics.per_class_precision["SUPPORTED"]
        print(f"Precision of SUPPORTED:")
        print(f"  FactChecker: {fc_sup_prec:.1%}")
        print(f"  Baseline:    {bl_sup_prec:.1%}")
        print(f"  Improvement: {(fc_sup_prec - bl_sup_prec):+.1%}")
    
    # Precision of REFUTED
    if "REFUTED" in fc_metrics.per_class_precision and "REFUTED" in bl_metrics.per_class_precision:
        fc_ref_prec = fc_metrics.per_class_precision["REFUTED"]
        bl_ref_prec = bl_metrics.per_class_precision["REFUTED"]
        print(f"\nPrecision of REFUTED:")
        print(f"  FactChecker: {fc_ref_prec:.1%}")
        print(f"  Baseline:    {bl_ref_prec:.1%}")
        print(f"  Improvement: {(fc_ref_prec - bl_ref_prec):+.1%}")
    
    # Recall of REFUTED
    if "REFUTED" in fc_metrics.per_class_recall and "REFUTED" in bl_metrics.per_class_recall:
        fc_ref_recall = fc_metrics.per_class_recall["REFUTED"]
        bl_ref_recall = bl_metrics.per_class_recall["REFUTED"]
        print(f"\nRecall of REFUTED:")
        print(f"  FactChecker: {fc_ref_recall:.1%}")
        print(f"  Baseline:    {bl_ref_recall:.1%}")
        print(f"  Improvement: {(fc_ref_recall - bl_ref_recall):+.1%}")
    
    print(f"{'='*60}")
    print(f"\nResults saved to: {results_file}")

    return results
