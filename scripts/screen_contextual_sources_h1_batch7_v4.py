"""Create the independent H1 source screen for contextual-source batch 7."""

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
INPUT_PATH = ROOT / "outputs" / "contextual_source_screening_input_batch7_v4.csv"
OUTPUT_PATH = (
    ROOT / "outputs" / "contextual_source_screening_H1_batch7_completed_v4.csv"
)
MANIFEST_PATH = (
    ROOT
    / "outputs"
    / "contextual_source_screening_H1_batch7_completed_v4.manifest.json"
)
H1_FIELDS = ["H1_decision", "H1_rationale"]


DECISIONS: dict[int, str] = {
    1: "exclude_not_relevant",
    2: "exclude_not_relevant",
    3: "exclude_not_relevant",
    4: "exclude_not_relevant",
    5: "exclude_not_relevant",
    6: "include_definition_or_review",
    7: "exclude_not_relevant",
    8: "exclude_not_relevant",
    9: "exclude_not_relevant",
    10: "exclude_not_relevant",
    11: "exclude_not_relevant",
    12: "exclude_not_relevant",
    13: "exclude_not_relevant",
    14: "exclude_not_relevant",
    15: "exclude_not_relevant",
    16: "exclude_not_relevant",
    17: "exclude_not_relevant",
    18: "exclude_not_relevant",
    19: "exclude_not_relevant",
    20: "exclude_not_relevant",
    21: "exclude_not_relevant",
    22: "exclude_not_relevant",
    23: "exclude_not_relevant",
    24: "include_definition_or_review",
    25: "exclude_not_relevant",
    26: "exclude_not_relevant",
    27: "exclude_not_relevant",
    28: "exclude_not_relevant",
    29: "exclude_not_relevant",
    30: "exclude_not_relevant",
    31: "exclude_not_relevant",
    32: "exclude_not_relevant",
    33: "include_definition_or_review",
    34: "exclude_not_relevant",
    35: "exclude_not_relevant",
    36: "exclude_not_relevant",
    37: "exclude_not_relevant",
    38: "exclude_not_relevant",
    39: "exclude_not_relevant",
    40: "exclude_not_relevant",
    41: "exclude_not_relevant",
    42: "exclude_not_relevant",
    43: "exclude_not_relevant",
    44: "exclude_not_relevant",
    45: "exclude_not_relevant",
    46: "exclude_not_relevant",
    47: "exclude_not_relevant",
    48: "exclude_not_relevant",
    49: "exclude_not_relevant",
    50: "exclude_not_relevant",
    51: "exclude_not_relevant",
    52: "exclude_not_relevant",
    53: "exclude_not_relevant",
    54: "exclude_not_relevant",
    55: "exclude_not_relevant",
    56: "exclude_not_relevant",
    57: "exclude_not_relevant",
    58: "exclude_not_relevant",
    59: "exclude_not_relevant",
    60: "exclude_not_relevant",
    61: "exclude_not_relevant",
    62: "exclude_not_relevant",
    63: "exclude_not_relevant",
    64: "exclude_not_relevant",
    65: "exclude_not_relevant",
    66: "exclude_not_relevant",
    67: "exclude_not_relevant",
    68: "exclude_not_relevant",
    69: "exclude_not_relevant",
    70: "exclude_not_relevant",
    71: "exclude_not_relevant",
    72: "exclude_not_relevant",
    73: "exclude_not_relevant",
    74: "exclude_not_relevant",
    75: "exclude_not_relevant",
    76: "exclude_not_relevant",
    77: "exclude_not_relevant",
    78: "exclude_not_relevant",
    79: "exclude_not_relevant",
    80: "exclude_not_relevant",
    81: "exclude_not_relevant",
    82: "exclude_not_relevant",
    83: "exclude_not_relevant",
    84: "exclude_not_relevant",
    85: "exclude_not_relevant",
    86: "exclude_not_relevant",
    87: "exclude_not_relevant",
    88: "exclude_not_relevant",
    89: "exclude_not_relevant",
    90: "include_definition_or_review",
    91: "include_definition_or_review",
    92: "exclude_not_relevant",
    93: "exclude_not_relevant",
    94: "exclude_not_relevant",
    95: "exclude_not_relevant",
    96: "exclude_not_relevant",
    97: "exclude_not_relevant",
    98: "exclude_not_relevant",
    99: "exclude_not_relevant",
    100: "exclude_not_relevant",
    101: "exclude_not_relevant",
    102: "exclude_not_relevant",
    103: "exclude_not_relevant",
    104: "exclude_not_relevant",
    105: "exclude_not_relevant",
    106: "exclude_not_relevant",
    107: "exclude_not_relevant",
    108: "exclude_not_relevant",
    109: "exclude_not_relevant",
    110: "uncertain",
    111: "exclude_not_relevant",
    112: "exclude_not_relevant",
    113: "include_definition_or_review",
    114: "exclude_not_relevant",
    115: "exclude_not_relevant",
    116: "exclude_not_relevant",
    117: "exclude_not_relevant",
    118: "exclude_not_relevant",
    119: "exclude_not_relevant",
    120: "exclude_not_relevant",
}


INCLUDE_REASONS: dict[int, str] = {
    6: "It directly introduces reference-list/keyword novelty metrics for research articles, a paper-level publication-time innovation construct.",
    24: "It reviews open-access indicators and methods for scholarly production, reach, and impact, including article-level scholarly communication measures.",
    33: "It is an article-level study of hybrid open-access metadata, a publication-time access attribute relevant to potential-impact modelling.",
    90: "It applies a publication-reference/topic analysis to measure interdisciplinarity of scholarly articles, a paper-level T0 knowledge-base construct.",
    91: "It explicitly predicts paper citations using only characteristics observable at double-blind submission, making it a direct potential-impact-at-T0 application.",
    113: "It operationalizes and empirically tests creative-scholarship indicators, including openness and idea density, for scholarly publications.",
}


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rationale_for(index: int, title: str, decision: str) -> str:
    """Create a concise blind-review rationale citing supplied English metadata."""
    evidence = f"English title evidence: “{title}.” "
    if decision == "include_definition_or_review":
        return evidence + INCLUDE_REASONS[index]
    if decision == "uncertain":
        return (
            evidence
            + "The title suggests an assessment model involving novelty and impact, but the available metadata "
            "does not establish that its unit and operation are a paper-level, publication-time indicator."
        )
    return (
        evidence
        + "The available title/abstract does not establish a paper-level innovation, quality, or T0 potential-impact "
        "indicator definition, application, validation, or review; domain maps, non-paper units, and future outcome "
        "measures are out of scope."
    )


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read the input CSV and retain source field order."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def validate_rows(
    source: list[dict[str, str]],
    completed: list[dict[str, str]],
    source_fields: list[str],
) -> None:
    """Verify frozen source fields and complete H1 annotations."""
    allowed = {"include_definition_or_review", "exclude_not_relevant", "uncertain"}
    assert len(source) == len(completed) == 120
    assert set(DECISIONS) == set(range(1, 121))
    for before, after in zip(source, completed, strict=True):
        for field in source_fields:
            assert before[field] == after[field], field
        assert after["H1_decision"] in allowed
        assert after["H1_rationale"]


def write_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write the source-preserving H1 annotation file."""
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Create independent batch-7 H1 annotations and their audit manifest."""
    source, source_fields = read_rows(INPUT_PATH)
    assert not (set(H1_FIELDS) & set(source_fields))
    completed = [dict(row) for row in source]
    for index, row in enumerate(completed, start=1):
        decision = DECISIONS[index]
        row["H1_decision"] = decision
        row["H1_rationale"] = rationale_for(index, row["title"], decision)
    validate_rows(source, completed, source_fields)
    fieldnames = [*source_fields, *H1_FIELDS]
    write_rows(completed, fieldnames)
    counts = Counter(row["H1_decision"] for row in completed)
    manifest: dict[str, Any] = {
        "schema": "contextual_source_screening_h1_batch7_manifest_v4",
        "reviewer": "H1",
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "output_sha256": sha256(OUTPUT_PATH),
        "source_count": len(completed),
        "h1_decision_counts": dict(sorted(counts.items())),
        "qwen_or_ollama_used": False,
        "read_ai_or_h2_or_old_batch_outputs": False,
        "frozen_input_fields_preserved": source_fields,
        "appended_h1_fields": H1_FIELDS,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
