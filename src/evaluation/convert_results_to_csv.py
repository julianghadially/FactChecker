"""Convert evaluation JSON results to CSV format.

Usage:
    python -m src.evaluation.convert_to_csv results/evaluation_20251209_161710.json
    python -m src.evaluation.convert_to_csv results/evaluation_20251209_161710.json --output results.csv
"""

import argparse
import json
import csv
from pathlib import Path


def convert_results_json_to_csv(json_path: str, output_path: str = None):
    """Convert evaluation JSON results to CSV.
    
    Args:
        json_path: Path to the evaluation JSON file.
        output_path: Optional output CSV path. If not provided, uses same name as JSON with .csv extension.
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    # Determine output path
    if output_path is None:
        output_path = json_path.with_suffix('.csv')
    else:
        output_path = Path(output_path)
    
    # Load JSON
    print(f"Loading JSON from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract detailed results
    if 'detailed_results' not in data:
        raise ValueError("JSON file does not contain 'detailed_results' field")
    
    detailed_results = data['detailed_results']
    
    if not detailed_results:
        print("Warning: No detailed results found in JSON file")
        return
    
    # Write CSV
    print(f"Writing CSV to: {output_path}")
    with open(f"results/{output_path}", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'claim',
            'ground_truth',
            'factchecker_prediction',
            'baseline_prediction'
        ])
        
        # Write rows
        for result in detailed_results:
            writer.writerow([
                result.get('claim', ''),
                result.get('ground_truth', ''),
                result.get('factchecker_prediction', ''),
                result.get('baseline_prediction', '')
            ])
    
    print(f"Successfully converted {len(detailed_results)} results to CSV")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Convert evaluation JSON results to CSV format"
    )
    parser.add_argument(
        'json_file',
        type=str,
        help='Path to the evaluation JSON file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output CSV file path (default: same as JSON with .csv extension)'
    )
    
    args = parser.parse_args()
    
    try:
        convert_results_json_to_csv(args.json_file, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

