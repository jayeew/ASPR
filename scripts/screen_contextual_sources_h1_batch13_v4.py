"""Create the independent H1 source-screening completion for batch 13."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

INPUT_PATH = Path(
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "contextual_source_screening_input_batch13_v4.csv"
)
OUTPUT_PATH = Path("outputs/contextual_source_screening_H1_batch13_completed_v4.csv")
MANIFEST_PATH = Path(
    "outputs/contextual_source_screening_H1_batch13_completed_v4.manifest.json"
)
H1_FIELDS = ("H1_decision", "H1_rationale", "H1_evidence_span")

H1_ASSESSMENTS: dict[str, tuple[str, str]] = {
    "openalex:w7199889110": (
        "exclude",
        (
            "The title and abstract describe a doctoral thesis on machine-learning "
            "evaluation of building energy performance. They do not indicate a study "
            "of innovation impacts or an eligible source for the contextual review."
        ),
    ),
    "openalex:w81854534": (
        "exclude",
        (
            "The title and abstract describe a bibliometric and social-network analysis "
            "of policy-network scholarship. They do not study innovation impacts or an "
            "eligible contextual source for the review."
        ),
    ),
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_span(row: dict[str, str]) -> str:
    """Build a concise, input-derived evidence span for the H1 assessment."""
    return f"Title: {row['title']} | Abstract: {row['abstract']}"


def read_input() -> tuple[list[str], list[dict[str, str]]]:
    """Read the frozen batch input without interpreting non-source-screen fields."""
    with INPUT_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Batch 13 input has no header.")
        return reader.fieldnames, list(reader)


def completed_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Add the independently assessed H1 fields to each frozen input row."""
    expected_keys = set(H1_ASSESSMENTS)
    observed_keys = {row["record_key"] for row in rows}
    if observed_keys != expected_keys:
        raise ValueError("Batch 13 record keys do not match the fixed H1 assessments.")

    completed: list[dict[str, str]] = []
    for row in rows:
        decision, rationale = H1_ASSESSMENTS[row["record_key"]]
        completed.append(
            {
                **row,
                "H1_decision": decision,
                "H1_rationale": rationale,
                "H1_evidence_span": evidence_span(row),
            }
        )
    return completed


def write_csv(fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write the completed CSV while preserving the frozen input columns."""
    output_fields = [*fieldnames, *H1_FIELDS]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(source_fields: list[str], rows: list[dict[str, str]]) -> None:
    """Write a provenance manifest for the blind H1 completion."""
    decisions = Counter(row["H1_decision"] for row in rows)
    manifest: dict[str, Any] = {
        "artifact": OUTPUT_PATH.name,
        "artifact_sha256": sha256(OUTPUT_PATH),
        "batch": 13,
        "blind_review_constraints": [
            "H1 decisions were made from this batch's title and abstract only.",
            "No AI, H2, prior-batch, Qwen, or Ollama results were consulted.",
        ],
        "decision_counts": dict(sorted(decisions.items())),
        "h1_fields": list(H1_FIELDS),
        "input": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "output_columns": [*source_fields, *H1_FIELDS],
        "source_rows": len(rows),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Create the batch 13 H1 screening artifact and its manifest."""
    fields, rows = read_input()
    if len(rows) != 2:
        raise ValueError(f"Expected 2 batch-13 rows, found {len(rows)}.")
    completed = completed_rows(rows)
    write_csv(fields, completed)
    write_manifest(fields, completed)


if __name__ == "__main__":
    main()
