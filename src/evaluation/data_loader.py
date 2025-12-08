"""Data loader for the HOVER dataset."""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, Optional


# Label schemas for different datasets
class LabelSchema:
    """Base class for dataset label schemas."""

    @classmethod
    def normalize_ground_truth(cls, label: str) -> str:
        """Normalize a ground truth label to internal format."""
        raise NotImplementedError

    @classmethod
    def normalize_prediction(cls, verdict: str) -> str:
        """Normalize a prediction to match ground truth format."""
        raise NotImplementedError

    @classmethod
    def get_labels(cls) -> list[str]:
        """Get list of valid labels for this schema."""
        raise NotImplementedError


class HoverLabelSchema(LabelSchema):
    """HOVER dataset uses SUPPORTED and NOT_SUPPORTED only."""

    # Mapping for ground truth labels
    GROUND_TRUTH_MAP = {
        "SUPPORTED": "SUPPORTED",
        "NOT_SUPPORTED": "NOT_SUPPORTED",
    }

    # Mapping for predictions (our system outputs)
    # Merge refuted into not_supported since HOVER doesn't distinguish
    PREDICTION_MAP = {
        # Pipeline aggregator outputs
        "SUPPORTED": "SUPPORTED",
        "CONTAINS_UNSUPPORTED_CLAIMS": "NOT_SUPPORTED",
        "CONTAINS_REFUTED_CLAIMS": "NOT_SUPPORTED",  # Merge into NOT_SUPPORTED
        # Individual claim verdicts (lowercase)
        "supported": "SUPPORTED",
        "not_supported": "NOT_SUPPORTED",
        "refuted": "NOT_SUPPORTED",  # Merge into NOT_SUPPORTED
        # Baseline model outputs
        "NOT_ENOUGH_INFO": "NOT_SUPPORTED",
        "REFUTED": "NOT_SUPPORTED",
        # Error handling
        "ERROR": "ERROR",
    }

    @classmethod
    def normalize_ground_truth(cls, label: str) -> str:
        return cls.GROUND_TRUTH_MAP.get(label, label)

    @classmethod
    def normalize_prediction(cls, verdict: str) -> str:
        return cls.PREDICTION_MAP.get(verdict, verdict)

    @classmethod
    def get_labels(cls) -> list[str]:
        return ["SUPPORTED", "NOT_SUPPORTED"]


class ThreeClassLabelSchema(LabelSchema):
    """Three-class schema: SUPPORTED, REFUTED, NOT_ENOUGH_INFO."""

    GROUND_TRUTH_MAP = {
        "SUPPORTED": "SUPPORTED",
        "REFUTED": "REFUTED",
        "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO",
    }

    PREDICTION_MAP = {
        # Pipeline aggregator outputs
        "SUPPORTED": "SUPPORTED",
        "CONTAINS_UNSUPPORTED_CLAIMS": "NOT_ENOUGH_INFO",
        "CONTAINS_REFUTED_CLAIMS": "REFUTED",
        # Individual claim verdicts (lowercase)
        "supported": "SUPPORTED",
        "not_supported": "NOT_ENOUGH_INFO",
        "refuted": "REFUTED",
        # Baseline model outputs
        "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO",
        "REFUTED": "REFUTED",
        # Error handling
        "ERROR": "ERROR",
    }

    @classmethod
    def normalize_ground_truth(cls, label: str) -> str:
        return cls.GROUND_TRUTH_MAP.get(label, label)

    @classmethod
    def normalize_prediction(cls, verdict: str) -> str:
        return cls.PREDICTION_MAP.get(verdict, verdict)

    @classmethod
    def get_labels(cls) -> list[str]:
        return ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


class FacToolLabelSchema(LabelSchema):
    """FacTool dataset uses true/false labels mapped to SUPPORTED/REFUTED."""

    GROUND_TRUTH_MAP = {
        "true": "SUPPORTED",
        "false": "REFUTED",
        "True": "SUPPORTED",
        "False": "REFUTED",
    }

    PREDICTION_MAP = {
        # Pipeline aggregator outputs
        "SUPPORTED": "SUPPORTED",
        "CONTAINS_UNSUPPORTED_CLAIMS": "UNKNOWN",  # Treat unsupported as refuted for FacTool
        "CONTAINS_REFUTED_CLAIMS": "REFUTED",
        # Individual claim verdicts (lowercase)
        "supported": "SUPPORTED",
        "not_supported": "UNKNOWN",  # Treat unsupported as refuted for FacTool
        "refuted": "REFUTED",
        # Baseline model outputs
        "NOT_ENOUGH_INFO": "UNKNOWN",  # Treat as refuted for FacTool
        "REFUTED": "REFUTED",
        # Error handling
        "ERROR": "ERROR",
    }

    @classmethod
    def normalize_ground_truth(cls, label: str) -> str:
        return cls.GROUND_TRUTH_MAP.get(label, label)

    @classmethod
    def normalize_prediction(cls, verdict: str) -> str:
        return cls.PREDICTION_MAP.get(verdict, verdict)

    @classmethod
    def get_labels(cls) -> list[str]:
        return ["SUPPORTED", "REFUTED"]


def detect_label_schema(labels: set[str]) -> type[LabelSchema]:
    """Auto-detect the label schema based on labels present in dataset."""
    # Check for FacTool format (true/false)
    if "true" in labels or "false" in labels or "True" in labels or "False" in labels:
        return FacToolLabelSchema
    elif "REFUTED" in labels or "NOT_ENOUGH_INFO" in labels:
        return ThreeClassLabelSchema
    elif "NOT_SUPPORTED" in labels:
        return HoverLabelSchema
    else:
        # Default to HOVER schema
        return HoverLabelSchema


@dataclass
class HoverExample:
    """A single example from the HOVER dataset."""

    uid: str
    claim: str
    label: str  # Actual label from dataset
    supporting_facts: list[tuple[str, int]]
    num_hops: int


@dataclass
class DatasetWithSchema:
    """Dataset with its associated label schema."""

    examples: list[HoverExample]
    schema: type[LabelSchema]


def load_dataset(
    path: str = "data/hover_dev_release_v1.1.json",
    limit: Optional[int] = None
) -> DatasetWithSchema:
    """Load dataset from JSON or JSONL file.

    Args:
        path: Path to the JSON/JSONL file (relative to project root).
        limit: Optional limit on number of examples to load.

    Returns:
        DatasetWithSchema containing examples and detected label schema.

    Raises:
        FileNotFoundError: If the dataset file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    # Check if JSONL format (one JSON object per line)
    is_jsonl = path.endswith(".jsonl")
    
    if is_jsonl:
        data = []
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Invalid JSON on line {line_num} of {path}: {e.msg}",
                        e.doc,
                        e.pos
                    )
    else:
        with open(file_path, "r") as f:
            data = json.load(f)

    # Detect label schema from data
    all_labels = set(item["label"] for item in data)
    schema = detect_label_schema(all_labels)
    print(f"Detected label schema: {schema.__name__}")
    print(f"Labels in dataset: {all_labels}")

    examples = []
    for idx, item in enumerate(data):
        # Handle different dataset formats
        uid = item.get("uid", f"example_{idx}")
        claim = item.get("claim", item.get("statement", ""))
        label = item["label"]
        
        # HOVER format has supporting_facts and num_hops, FacTool doesn't
        supporting_facts = item.get("supporting_facts", [])
        num_hops = item.get("num_hops", 0)
        
        examples.append(HoverExample(
            uid=uid,
            claim=claim,
            label=label,
            supporting_facts=[(sf[0], sf[1]) if isinstance(sf, (list, tuple)) and len(sf) >= 2 else ("", 0) for sf in supporting_facts],
            num_hops=num_hops
        ))

    if limit:
        examples = examples[:limit]

    return DatasetWithSchema(examples=examples, schema=schema)
