"""Create the blinded H1 contextual-source screen for evidence-derived v4 batch 5."""

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
INPUT_PATH = ROOT / "outputs" / "contextual_source_screening_input_batch5_v4.csv"
OUTPUT_PATH = (
    ROOT / "outputs" / "contextual_source_screening_H1_batch5_completed_v4.csv"
)
MANIFEST_PATH = (
    ROOT
    / "outputs"
    / "contextual_source_screening_H1_batch5_completed_v4.manifest.json"
)
EDITABLE_FIELDS = {"screen_decision", "evidence_span", "rationale"}


DECISIONS: dict[int, str] = {
    1: "exclude_not_relevant",
    2: "exclude_not_relevant",
    3: "include_definition_or_review",
    4: "include_definition_or_review",
    5: "include_definition_or_review",
    6: "include_definition_or_review",
    7: "include_definition_or_review",
    8: "exclude_not_relevant",
    9: "exclude_not_relevant",
    10: "include_definition_or_review",
    11: "include_definition_or_review",
    12: "include_definition_or_review",
    13: "include_definition_or_review",
    14: "include_definition_or_review",
    15: "exclude_not_relevant",
    16: "exclude_not_relevant",
    17: "exclude_not_relevant",
    18: "exclude_not_relevant",
    19: "exclude_not_relevant",
    20: "exclude_not_relevant",
    21: "exclude_not_relevant",
    22: "exclude_not_relevant",
    23: "include_definition_or_review",
    24: "include_definition_or_review",
    25: "include_definition_or_review",
    26: "include_definition_or_review",
    27: "exclude_not_relevant",
    28: "uncertain",
    29: "exclude_not_relevant",
    30: "exclude_not_relevant",
    31: "exclude_not_relevant",
    32: "exclude_not_relevant",
    33: "include_definition_or_review",
    34: "include_definition_or_review",
    35: "include_definition_or_review",
    36: "exclude_not_relevant",
    37: "exclude_not_relevant",
    38: "exclude_not_relevant",
    39: "include_definition_or_review",
    40: "include_definition_or_review",
    41: "include_definition_or_review",
    42: "exclude_not_relevant",
    43: "include_definition_or_review",
    44: "include_definition_or_review",
    45: "include_definition_or_review",
    46: "exclude_not_relevant",
    47: "include_definition_or_review",
    48: "exclude_not_relevant",
    49: "exclude_not_relevant",
    50: "include_definition_or_review",
    51: "include_definition_or_review",
    52: "exclude_not_relevant",
    53: "include_definition_or_review",
    54: "include_definition_or_review",
    55: "exclude_not_relevant",
    56: "exclude_not_relevant",
    57: "include_definition_or_review",
    58: "exclude_not_relevant",
    59: "include_definition_or_review",
    60: "exclude_not_relevant",
    61: "include_definition_or_review",
    62: "exclude_not_relevant",
    63: "include_definition_or_review",
    64: "include_definition_or_review",
    65: "include_definition_or_review",
    66: "exclude_not_relevant",
    67: "exclude_not_relevant",
    68: "exclude_not_relevant",
    69: "exclude_not_relevant",
    70: "exclude_not_relevant",
    71: "exclude_not_relevant",
    72: "include_definition_or_review",
    73: "include_definition_or_review",
    74: "exclude_not_relevant",
    75: "exclude_not_relevant",
    76: "exclude_not_relevant",
    77: "exclude_not_relevant",
    78: "include_definition_or_review",
    79: "include_definition_or_review",
    80: "exclude_not_relevant",
    81: "exclude_not_relevant",
    82: "exclude_not_relevant",
    83: "include_definition_or_review",
    84: "exclude_not_relevant",
    85: "include_definition_or_review",
    86: "include_definition_or_review",
    87: "include_definition_or_review",
    88: "include_definition_or_review",
    89: "include_definition_or_review",
    90: "include_definition_or_review",
    91: "exclude_not_relevant",
    92: "exclude_not_relevant",
    93: "exclude_not_relevant",
    94: "include_definition_or_review",
    95: "include_definition_or_review",
    96: "include_definition_or_review",
    97: "include_definition_or_review",
    98: "exclude_not_relevant",
    99: "exclude_not_relevant",
    100: "exclude_not_relevant",
    101: "include_definition_or_review",
    102: "include_definition_or_review",
    103: "include_definition_or_review",
    104: "include_definition_or_review",
    105: "include_definition_or_review",
    106: "include_definition_or_review",
    107: "include_definition_or_review",
    108: "uncertain",
    109: "include_definition_or_review",
    110: "exclude_not_relevant",
    111: "exclude_not_relevant",
    112: "exclude_not_relevant",
    113: "exclude_not_relevant",
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
    """Explain the screening result using only the supplied metadata."""
    if decision == "include_definition_or_review":
        return (
            "Worth restoring: the title/abstract directly concerns scholarly impact or evaluation, "
            "or defines, applies, validates, or reviews a publication-time paper attribute (for example "
            "knowledge combination, interdisciplinarity, access, collaboration, text, or attention) "
            "that can support a paper-level innovation or potential-impact construct."
        )
    if decision == "uncertain":
        return (
            "Potentially relevant because it concerns a publication-time content or authorship feature, "
            "but the available title/abstract does not establish a direct paper-level innovation or "
            "potential-impact definition, application, validation, or review."
        )
    return (
        "Exclude: the available metadata describes a substantive-domain literature map, general research "
        "practice, or other topic without a direct original definition, application, validation, or review "
        "of a paper-level innovation or publication-time potential-impact construct."
    )


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read rows and CSV field order from the supplied input."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def validate_rows(
    original: list[dict[str, str]],
    completed: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    """Ensure only review fields differ and all values are valid."""
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
    """Write the completed screen while preserving the original schema."""
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Apply independent H1 decisions and emit a verification manifest."""
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
        "schema": "contextual_source_screening_h1_batch5_manifest_v4",
        "reviewer": "H1",
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "output_sha256": sha256(OUTPUT_PATH),
        "source_count": len(completed),
        "screen_decision_counts": dict(sorted(counts.items())),
        "qwen_or_ollama_used": False,
        "read_ai_or_h2_or_other_batch5_outputs": False,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
