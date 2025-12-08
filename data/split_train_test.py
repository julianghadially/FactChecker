"""Script to randomly split a JSONL file into train and test sets."""

import json
import random
from pathlib import Path


def split_jsonl(
    input_file: str,
    train_file: str,
    test_file: str,
    test_ratio: float = 0.2,
    seed: int = 42
):
    """Split a JSONL file into train and test sets.
    
    Args:
        input_file: Path to input JSONL file.
        train_file: Path to output train JSONL file.
        test_file: Path to output test JSONL file.
        test_ratio: Proportion of data to use for test (default: 0.2 = 20%).
        seed: Random seed for reproducibility (default: 42).
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read all lines from input file
    lines = []
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Reading {input_file}...")
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                lines.append(line)
    
    total_lines = len(lines)
    print(f"Total lines: {total_lines}")
    
    # Shuffle lines randomly
    random.shuffle(lines)
    
    # Calculate split point
    test_size = int(total_lines * test_ratio)
    train_size = total_lines - test_size
    
    # Split into train and test
    train_lines = lines[:train_size]
    test_lines = lines[train_size:]
    
    print(f"Train set: {train_size} lines ({train_size/total_lines:.1%})")
    print(f"Test set: {test_size} lines ({test_size/total_lines:.1%})")
    
    # Write train file
    train_path = Path(train_file)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Writing train set to {train_file}...")
    with open(train_path, "w") as f:
        for line in train_lines:
            f.write(line + "\n")
    
    # Write test file
    test_path = Path(test_file)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Writing test set to {test_file}...")
    with open(test_path, "w") as f:
        for line in test_lines:
            f.write(line + "\n")
    
    print("Split complete!")
    
    # Verify the split
    print("\nVerifying split...")
    with open(train_path, "r") as f:
        train_count = sum(1 for line in f if line.strip())
    with open(test_path, "r") as f:
        test_count = sum(1 for line in f if line.strip())
    
    print(f"Train file has {train_count} lines")
    print(f"Test file has {test_count} lines")
    print(f"Total: {train_count + test_count} lines (expected {total_lines})")


if __name__ == "__main__":
    # Hardcoded file paths
    input_file = "FacTool_QA.jsonl"
    train_file = "FacTool_QA_train.jsonl"
    test_file = "FacTool_QA_test.jsonl"
    
    # Default parameters
    test_ratio = 0.5 
    seed = 42  # For reproducibility
    
    # Get the directory where this script is located (data folder)
    script_dir = Path(__file__).parent
    
    # Construct full paths
    input_path = script_dir / input_file
    train_path = script_dir / train_file
    test_path = script_dir / test_file
    
    # Run the split
    split_jsonl(
        input_file=str(input_path),
        train_file=str(train_path),
        test_file=str(test_path),
        test_ratio=test_ratio,
        seed=seed
    )

