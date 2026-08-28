"""Evaluation script comparing FactChecker vs Baseline on HOVER dataset."""

import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import dspy

from .data_loader import load_dataset, load_csv_dataset
from .metrics import calculate_metrics, print_metrics


def run_evaluation(
    fact_checker,
    baseline_model,
    sample_size: int = 100,
    output_dir: str = "results",
    dataset_path: str = "data/FactChecker_news_claims.csv",
    num_threads: int = 5 #firecrawl concurrency limit is 5
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
    if dataset_path.endswith(".json") or dataset_path.endswith(".jsonl"):
        dataset_with_schema = load_dataset(path=dataset_path, limit=sample_size)
        dataset = dataset_with_schema.examples
        schema = dataset_with_schema.schema
    elif dataset_path.endswith(".csv"):
        dataset_with_schema = load_csv_dataset(path=dataset_path, limit=sample_size)
        dataset = dataset_with_schema.examples
        schema = dataset_with_schema.schema
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")
    print(f"\nUsing label schema: {schema.__name__}")
    print(f"Valid labels: {schema.get_labels()}")

    # Define accuracy metric using the detected schema
    def accuracy_metric(example: dspy.Example, prediction) -> float:
        """Simple accuracy metric for dspy.Evaluate using the detected schema."""
        # Extract ground truth label
        gold_label = example.label
        
        # Extract prediction label (FactChecker has overall_verdict, Baseline has verdict in dict)
        if hasattr(prediction, 'overall_verdict') and prediction.overall_verdict is not None:
            pred_label = prediction.overall_verdict
        elif isinstance(prediction, dict):
            pred_label = prediction.get('verdict', 'ERROR')
        else:
            pred_label = str(prediction) if prediction is not None else 'ERROR'
        
        # Normalize both labels using the detected schema
        normalized_gold = schema.normalize_ground_truth(gold_label)
        normalized_pred = schema.normalize_prediction(pred_label)
        
        # Return 1.0 if match, 0.0 otherwise
        return 1.0 if normalized_gold == normalized_pred else 0.0

    # Collect predictions
    fc_predictions = []
    baseline_predictions = []
    ground_truth = []
    detailed_results = []

    examples = [
        dspy.Example(statement=ex.claim, label=ex.label).with_inputs("statement")
        for ex in dataset
    ]

    # Create evaluator with devset and accuracy metric
    evaluator = dspy.Evaluate(
        devset=examples,  # Required keyword argument
        metric=accuracy_metric,  # Accuracy metric function
        num_threads=num_threads,
        display_progress=True,
        disable_cache=True
    )

    # Call evaluator with the program
    print("Evaluating FactChecker with threading...")
    fc_result = evaluator(fact_checker)  # Returns EvaluationResult

    # Extract predictions from result.results (list of (example, prediction, score) tuples)
    fc_predictions = [
        pred.overall_verdict if hasattr(pred, 'overall_verdict') else str(pred)
        for _, pred, _ in fc_result.results
    ]

    # Same for baseline
    print("Evaluating Baseline with threading...")
    baseline_result = evaluator(baseline_model)
    baseline_predictions = [
        pred["verdict"] if isinstance(pred, dict) else (pred.verdict if hasattr(pred, 'verdict') else str(pred))
        for _, pred, _ in baseline_result.results
    ]


    ground_truth = [ex.label for ex in dataset]

    detailed_results = []
    for i, example in enumerate(dataset):
        detailed_results.append({
            "uid": example.uid,
            "claim": example.claim,
            "ground_truth": example.label,
            "factchecker_prediction": fc_predictions[i],
            "factchecker_normalized": schema.normalize_prediction(fc_predictions[i]),
            "baseline_prediction": baseline_predictions[i],
            "baseline_normalized": schema.normalize_prediction(baseline_predictions[i]),
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
            "accuracy_on_predictions": fc_metrics.accuracy_on_predictions,
            "total_examples": fc_metrics.total_examples,
            "valid_examples": fc_metrics.valid_examples,
            "error_count": fc_metrics.error_count,
            "per_class_precision": fc_metrics.per_class_precision,
            "per_class_recall": fc_metrics.per_class_recall,
            "confusion_matrix": fc_metrics.confusion_matrix
        },
        "baseline": {
            "accuracy": bl_metrics.accuracy,
            "accuracy_on_predictions": bl_metrics.accuracy_on_predictions,
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
    improvement_on_predictions = fc_metrics.accuracy_on_predictions - bl_metrics.accuracy_on_predictions
    print(f"\nAccuracy Improvement: {improvement:+.1%}")
    print(f"Accuracy on Predictions Improvement: {improvement_on_predictions:+.1%}")
    
    # Print key metrics comparison
    print(f"\n{'='*60}")
    print("KEY METRICS COMPARISON")
    print(f"{'='*60}")
    
    # Accuracy on Predictions
    print(f"Accuracy on Predictions:")
    print(f"  FactChecker: {fc_metrics.accuracy_on_predictions:.1%}")
    print(f"  Baseline:    {bl_metrics.accuracy_on_predictions:.1%}")
    print(f"  Improvement: {improvement_on_predictions:+.1%}")
    print()
    
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
