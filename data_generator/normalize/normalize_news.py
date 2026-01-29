#!/usr/bin/env python3
"""Convert FactChecker_news_claims.csv: normalize ground truth labels and rename claim -> statement.

Output remains CSV. Run from project root:
    python -m data_generator.convert_data_format
    python -m data_generator.convert_data_format --input data/FactChecker_news_claims.csv --output data/FactChecker_news_claims_normalized.csv
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

# FacTool-style: TRUE/FALSE -> SUPPORTED/REFUTED (same as FacToolLabelSchema)
def _normalize_label(raw: str) -> str:
    key = (raw or "").strip().lower()
    return {"true": "SUPPORTED", "false": "REFUTED"}.get(key, (raw or "").strip())


def convert_csv(
    input_path: str = "data/FactChecker_news_claims.csv",
    output_path: Optional[str] = None,
) -> None:
    """Read CSV, normalize labels (TRUE/FALSE -> SUPPORTED/REFUTED), rename claim -> statement, write CSV."""
    input_path = Path(input_path)
    output_path = Path(output_path or input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows: List[Dict[str, str]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames_in = reader.fieldnames or []
        for row in reader:
            raw_label = (row.get("label") or "").strip()
            normalized_label = _normalize_label(raw_label)
            # Rename claim -> statement; keep other columns
            new_row = {}
            for k, v in row.items():
                if k == "claim":
                    new_row["statement"] = v or ""
                else:
                    new_row[k] = v or ""
            new_row["label"] = normalized_label
            rows.append(new_row)

    # Output columns: same as input but claim -> statement
    out_fieldnames = [
        "statement" if name == "claim" else name
        for name in fieldnames_in
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert FactChecker CSV: normalize labels, rename claim to statement."
    )
    parser.add_argument(
        "--input",
        default="data/FactChecker_news_claims.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: overwrite input)",
    )
    args = parser.parse_args()
    convert_csv(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
