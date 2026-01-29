#!/usr/bin/env python3
"""Convert FacTool_QA JSONL: normalize ground truth labels and rename claim -> statement.

Output remains JSONL. Run from project root:
    python -m data_generator.normalize.normalize_factool
    python -m data_generator.normalize.normalize_factool --input data/FacTool_QA_train.jsonl --output data/FacTool_QA_train_normalized.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# FacTool-style: true/false -> SUPPORTED/REFUTED (same as FacToolLabelSchema)
def _normalize_label(raw: str) -> str:
    key = (raw or "").strip().lower()
    return {"true": "SUPPORTED", "false": "REFUTED"}.get(key, (raw or "").strip())


def convert_jsonl(
    input_path: str = "data/FacTool_QA_train.jsonl",
    output_path: Optional[str] = None,
) -> None:
    """Read JSONL, normalize labels (true/false -> SUPPORTED/REFUTED), rename claim -> statement, write JSONL."""
    input_path = Path(input_path)
    output_path = Path(output_path or input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {line_num} of {input_path}: {e.msg}",
                    e.doc,
                    e.pos,
                ) from e

            raw_label = (obj.get("label") or "").strip() if isinstance(obj.get("label"), str) else str(obj.get("label", ""))
            normalized_label = _normalize_label(raw_label)

            # Rename claim -> statement; keep all other keys; set normalized label
            new_row: Dict[str, Any] = {}
            for k, v in obj.items():
                if k == "claim":
                    new_row["statement"] = v
                elif k == "label":
                    new_row["label"] = normalized_label
                else:
                    new_row[k] = v
            rows.append(new_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert FacTool QA JSONL: normalize labels, rename claim to statement."
    )
    parser.add_argument(
        "--input",
        default="data/FacTool_QA_train.jsonl",
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: overwrite input)",
    )
    args = parser.parse_args()
    convert_jsonl(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
