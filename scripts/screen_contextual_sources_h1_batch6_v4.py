"""Create the blinded H1 contextual-source screen for evidence-derived v4 batch 6."""

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
INPUT_PATH = ROOT / "outputs" / "contextual_source_screening_input_batch6_v4.csv"
OUTPUT_PATH = (
    ROOT / "outputs" / "contextual_source_screening_H1_batch6_completed_v4.csv"
)
MANIFEST_PATH = (
    ROOT
    / "outputs"
    / "contextual_source_screening_H1_batch6_completed_v4.manifest.json"
)
EDITABLE_FIELDS = {"screen_decision", "evidence_span", "rationale"}


DECISIONS: dict[int, str] = {
    1: "exclude_not_relevant",
    2: "exclude_not_relevant",
    3: "exclude_not_relevant",
    4: "exclude_not_relevant",
    5: "exclude_not_relevant",
    6: "exclude_not_relevant",
    7: "exclude_not_relevant",
    8: "include_definition_or_review",
    9: "exclude_not_relevant",
    10: "exclude_not_relevant",
    11: "include_definition_or_review",
    12: "include_definition_or_review",
    13: "include_definition_or_review",
    14: "include_definition_or_review",
    15: "include_definition_or_review",
    16: "include_definition_or_review",
    17: "include_definition_or_review",
    18: "include_definition_or_review",
    19: "exclude_not_relevant",
    20: "include_definition_or_review",
    21: "exclude_not_relevant",
    22: "include_definition_or_review",
    23: "include_definition_or_review",
    24: "include_definition_or_review",
    25: "include_definition_or_review",
    26: "exclude_not_relevant",
    27: "include_definition_or_review",
    28: "include_definition_or_review",
    29: "exclude_not_relevant",
    30: "exclude_not_relevant",
    31: "include_definition_or_review",
    32: "exclude_not_relevant",
    33: "exclude_not_relevant",
    34: "exclude_not_relevant",
    35: "exclude_not_relevant",
    36: "exclude_not_relevant",
    37: "include_definition_or_review",
    38: "exclude_not_relevant",
    39: "exclude_not_relevant",
    40: "include_definition_or_review",
    41: "exclude_not_relevant",
    42: "include_definition_or_review",
    43: "exclude_not_relevant",
    44: "exclude_not_relevant",
    45: "include_definition_or_review",
    46: "exclude_not_relevant",
    47: "include_definition_or_review",
    48: "exclude_not_relevant",
    49: "exclude_not_relevant",
    50: "include_definition_or_review",
    51: "exclude_not_relevant",
    52: "exclude_not_relevant",
    53: "exclude_not_relevant",
    54: "exclude_not_relevant",
    55: "include_definition_or_review",
    56: "exclude_not_relevant",
    57: "include_definition_or_review",
    58: "exclude_not_relevant",
    59: "include_definition_or_review",
    60: "include_definition_or_review",
    61: "include_definition_or_review",
    62: "include_definition_or_review",
    63: "exclude_not_relevant",
    64: "include_definition_or_review",
    65: "exclude_not_relevant",
    66: "include_definition_or_review",
    67: "exclude_not_relevant",
    68: "exclude_not_relevant",
    69: "include_definition_or_review",
    70: "include_definition_or_review",
    71: "include_definition_or_review",
    72: "include_definition_or_review",
    73: "include_definition_or_review",
    74: "include_definition_or_review",
    75: "include_definition_or_review",
    76: "include_definition_or_review",
    77: "include_definition_or_review",
    78: "exclude_not_relevant",
    79: "exclude_not_relevant",
    80: "include_definition_or_review",
    81: "exclude_not_relevant",
    82: "include_definition_or_review",
    83: "exclude_not_relevant",
    84: "include_definition_or_review",
    85: "exclude_not_relevant",
    86: "uncertain",
    87: "include_definition_or_review",
    88: "exclude_not_relevant",
    89: "include_definition_or_review",
    90: "include_definition_or_review",
    91: "include_definition_or_review",
    92: "exclude_not_relevant",
    93: "include_definition_or_review",
    94: "include_definition_or_review",
    95: "exclude_not_relevant",
    96: "exclude_not_relevant",
    97: "exclude_not_relevant",
    98: "exclude_not_relevant",
    99: "exclude_not_relevant",
    100: "exclude_not_relevant",
    101: "include_definition_or_review",
    102: "exclude_not_relevant",
    103: "exclude_not_relevant",
    104: "exclude_not_relevant",
    105: "exclude_not_relevant",
    106: "exclude_not_relevant",
    107: "exclude_not_relevant",
    108: "exclude_not_relevant",
    109: "exclude_not_relevant",
    110: "include_definition_or_review",
    111: "exclude_not_relevant",
    112: "include_definition_or_review",
    113: "include_definition_or_review",
    114: "exclude_not_relevant",
    115: "exclude_not_relevant",
    116: "exclude_not_relevant",
    117: "exclude_not_relevant",
    118: "exclude_not_relevant",
    119: "exclude_not_relevant",
    120: "exclude_not_relevant",
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rationale_for(decision: str) -> str:
    """Explain an H1 result from the supplied title and abstract."""
    if decision == "include_definition_or_review":
        return (
            "Worth restoring: the title/abstract defines, applies, validates, or reviews scholarly "
            "impact/evaluation, or a publication-time paper attribute (text, references, knowledge "
            "combination, access, collaboration, attention, or editorial process) that can support a "
            "paper-level innovation or potential-impact T0 construct."
        )
    if decision == "uncertain":
        return (
            "Potentially useful for publication-integrity context, but the available metadata does not "
            "establish a direct original definition, application, validation, or review of a paper-level "
            "innovation or publication-time potential-impact construct."
        )
    return (
        "Exclude: the available metadata describes a substantive-domain review/map or another topic that "
        "does not directly define, apply, validate, or review a paper-level innovation or publication-time "
        "potential-impact T0 construct."
    )


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read all CSV rows while retaining schema order."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def validate_rows(
    original: list[dict[str, str]],
    completed: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    """Verify row count, allowed decisions, evidence, and frozen fields."""
    allowed = {"include_definition_or_review", "exclude_not_relevant", "uncertain"}
    assert len(original) == len(completed) == 120
    assert set(DECISIONS) == set(range(1, 121))
    for before, after in zip(original, completed, strict=True):
        for field in fieldnames:
            if field not in EDITABLE_FIELDS:
                assert before[field] == after[field], field
        assert after["screen_decision"] in allowed
        assert after["evidence_span"] == after["title"]
        assert after["rationale"]


def write_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write the completed screening CSV without schema changes."""
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Apply independent H1 decisions and write a verification manifest."""
    original, fieldnames = read_rows(INPUT_PATH)
    assert EDITABLE_FIELDS.issubset(fieldnames)
    completed = [dict(row) for row in original]
    for index, row in enumerate(completed, start=1):
        decision = DECISIONS[index]
        row["screen_decision"] = decision
        row["evidence_span"] = row["title"]
        row["rationale"] = rationale_for(decision)
    validate_rows(original, completed, fieldnames)
    write_rows(completed, fieldnames)
    counts = Counter(row["screen_decision"] for row in completed)
    manifest: dict[str, Any] = {
        "schema": "contextual_source_screening_h1_batch6_manifest_v4",
        "reviewer": "H1",
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "output_sha256": sha256(OUTPUT_PATH),
        "source_count": len(completed),
        "screen_decision_counts": dict(sorted(counts.items())),
        "qwen_or_ollama_used": False,
        "read_ai_or_h2_or_other_batch6_outputs": False,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
