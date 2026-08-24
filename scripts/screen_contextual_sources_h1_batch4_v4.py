"""Create the blinded H1 contextual-source screen for evidence-derived v4 batch 4."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = (
    Path(__file__).resolve().parents[1]
    / "innovation_impact_feature_selection"
    / "evidence_derived_v4_rebuild"
)
INPUT_PATH = ROOT / "outputs" / "contextual_source_screening_input_batch4_v4.csv"
OUTPUT_PATH = (
    ROOT / "outputs" / "contextual_source_screening_H1_batch4_completed_v4.csv"
)
MANIFEST_PATH = (
    ROOT
    / "outputs"
    / "contextual_source_screening_H1_batch4_completed_v4.manifest.json"
)
EDITABLE_FIELDS = {"screen_decision", "evidence_span", "rationale"}


DECISIONS: dict[int, str] = {
    1: "include_definition_or_review",
    2: "exclude_not_relevant",
    3: "include_definition_or_review",
    4: "exclude_not_relevant",
    5: "include_definition_or_review",
    6: "exclude_not_relevant",
    7: "exclude_not_relevant",
    8: "include_definition_or_review",
    9: "include_definition_or_review",
    10: "include_definition_or_review",
    11: "include_definition_or_review",
    12: "include_definition_or_review",
    13: "exclude_not_relevant",
    14: "include_definition_or_review",
    15: "include_definition_or_review",
    16: "exclude_not_relevant",
    17: "exclude_not_relevant",
    18: "exclude_not_relevant",
    19: "exclude_not_relevant",
    20: "include_definition_or_review",
    21: "exclude_not_relevant",
    22: "exclude_not_relevant",
    23: "exclude_not_relevant",
    24: "include_definition_or_review",
    25: "exclude_not_relevant",
    26: "exclude_not_relevant",
    27: "include_definition_or_review",
    28: "exclude_not_relevant",
    29: "include_definition_or_review",
    30: "exclude_not_relevant",
    31: "exclude_not_relevant",
    32: "exclude_not_relevant",
    33: "include_definition_or_review",
    34: "include_definition_or_review",
    35: "exclude_not_relevant",
    36: "exclude_not_relevant",
    37: "exclude_not_relevant",
    38: "exclude_not_relevant",
    39: "include_definition_or_review",
    40: "exclude_not_relevant",
    41: "exclude_not_relevant",
    42: "exclude_not_relevant",
    43: "exclude_not_relevant",
    44: "exclude_not_relevant",
    45: "include_definition_or_review",
    46: "include_definition_or_review",
    47: "exclude_not_relevant",
    48: "include_definition_or_review",
    49: "exclude_not_relevant",
    50: "include_definition_or_review",
    51: "exclude_not_relevant",
    52: "exclude_not_relevant",
    53: "exclude_not_relevant",
    54: "include_definition_or_review",
    55: "exclude_not_relevant",
    56: "include_definition_or_review",
    57: "exclude_not_relevant",
    58: "exclude_not_relevant",
    59: "include_definition_or_review",
    60: "include_definition_or_review",
    61: "exclude_not_relevant",
    62: "include_definition_or_review",
    63: "include_definition_or_review",
    64: "include_definition_or_review",
    65: "include_definition_or_review",
    66: "exclude_not_relevant",
    67: "include_definition_or_review",
    68: "exclude_not_relevant",
    69: "exclude_not_relevant",
    70: "include_definition_or_review",
    71: "exclude_not_relevant",
    72: "include_definition_or_review",
    73: "include_definition_or_review",
    74: "exclude_not_relevant",
    75: "include_definition_or_review",
    76: "exclude_not_relevant",
    77: "include_definition_or_review",
    78: "include_definition_or_review",
    79: "exclude_not_relevant",
    80: "exclude_not_relevant",
    81: "include_definition_or_review",
    82: "include_definition_or_review",
    83: "exclude_not_relevant",
    84: "exclude_not_relevant",
    85: "exclude_not_relevant",
    86: "include_definition_or_review",
    87: "include_definition_or_review",
    88: "include_definition_or_review",
    89: "exclude_not_relevant",
    90: "exclude_not_relevant",
    91: "include_definition_or_review",
    92: "exclude_not_relevant",
    93: "include_definition_or_review",
    94: "exclude_not_relevant",
    95: "exclude_not_relevant",
    96: "exclude_not_relevant",
    97: "exclude_not_relevant",
    98: "exclude_not_relevant",
    99: "exclude_not_relevant",
    100: "include_definition_or_review",
    101: "include_definition_or_review",
    102: "exclude_not_relevant",
    103: "exclude_not_relevant",
    104: "exclude_not_relevant",
    105: "include_definition_or_review",
    106: "exclude_not_relevant",
    107: "include_definition_or_review",
    108: "include_definition_or_review",
    109: "exclude_not_relevant",
    110: "include_definition_or_review",
    111: "exclude_not_relevant",
    112: "exclude_not_relevant",
    113: "uncertain",
    114: "exclude_not_relevant",
    115: "include_definition_or_review",
    116: "include_definition_or_review",
    117: "include_definition_or_review",
    118: "include_definition_or_review",
    119: "include_definition_or_review",
    120: "include_definition_or_review",
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rationale_for(decision: str) -> str:
    """Give a concise H1 screening rationale without introducing external evidence."""
    if decision == "include_definition_or_review":
        return (
            "Worth retaining as a contextual source: the title/abstract directly addresses "
            "scholarly impact, publication/citation or alternative-metric measurement, or a "
            "paper-level attribute that can inform an original definition, application, validation, or review."
        )
    if decision == "uncertain":
        return (
            "The title suggests a potentially transferable diversity measurement method, but the "
            "available title/abstract does not establish that it defines or validates a scholarly-paper "
            "innovation or publication-time impact indicator."
        )
    return (
        "Exclude: this is a domain-specific bibliometric map, substantive-topic review, or general "
        "methods study rather than a source defining, applying, validating, or reviewing paper-level "
        "innovation or publication-time impact indicators."
    )


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read the input CSV preserving its schema and rows."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def validate_rows(
    original: list[dict[str, str]],
    completed: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    """Confirm frozen fields, allowed decisions, and non-empty H1 evidence."""
    assert len(original) == len(completed) == 120
    assert set(DECISIONS) == set(range(1, 121))
    for before, after in zip(original, completed, strict=True):
        for field in fieldnames:
            if field not in EDITABLE_FIELDS:
                assert before[field] == after[field], field
        assert after["screen_decision"] in {
            "include_definition_or_review",
            "exclude_not_relevant",
            "uncertain",
        }
        assert after["evidence_span"] == after["title"]
        assert after["rationale"]


def write_csv(rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write the completed CSV with stable UTF-8 CSV settings."""
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Apply independent H1 decisions and write the audit manifest."""
    original, fieldnames = read_rows(INPUT_PATH)
    assert set(EDITABLE_FIELDS).issubset(fieldnames)
    completed = [dict(row) for row in original]
    for index, row in enumerate(completed, start=1):
        decision = DECISIONS[index]
        row["screen_decision"] = decision
        row["evidence_span"] = row["title"]
        row["rationale"] = rationale_for(decision)
    validate_rows(original, completed, fieldnames)
    write_csv(completed, fieldnames)
    counts = Counter(row["screen_decision"] for row in completed)
    manifest: dict[str, Any] = {
        "schema": "contextual_source_screening_h1_batch4_manifest_v4",
        "reviewer": "H1",
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "output_sha256": sha256(OUTPUT_PATH),
        "source_count": len(completed),
        "screen_decision_counts": dict(sorted(counts.items())),
        "qwen_or_ollama_used": False,
        "read_ai_or_h2_or_other_batch4_outputs": False,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
