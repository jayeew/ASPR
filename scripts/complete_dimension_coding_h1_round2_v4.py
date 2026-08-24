"""Complete the independent H1 round-2 dimension coding for count families."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Final

ROOT: Final = Path("/home/jayee/workspace/ASPR")
INPUT_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "dimension_coding_H1_round2_v4.csv"
)
AUDIT_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "operational_equivalence_audit_v4.json"
)
OUTPUT_PATH: Final = ROOT / "outputs/dimension_coding_H1_round2_completed_v4.csv"
MANIFEST_PATH: Final = (
    ROOT / "outputs/dimension_coding_H1_round2_completed_v4.manifest.json"
)
EDITABLE_FIELDS: Final = [
    "dimension_label",
    "dimension_definition",
    "construct_role",
    "information_source",
    "t0_boundary",
    "bias_risk",
    "decision",
    "reason",
]


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a UTF-8 CSV and retain its field order."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    """Write records without changing the input schema."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


CODINGS: Final[dict[str, dict[str, str]]] = {
    "author_count": {
        "dimension_label": "Team composition and collaboration scale",
        "dimension_definition": (
            "The count of listed authors on a focal paper at publication, representing "
            "the scale of the producing team as a contextual control rather than an "
            "innovation or impact outcome."
        ),
        "construct_role": "context_control",
        "information_source": (
            "Focal-paper author byline, represented by audited publication-time "
            "OpenAlex author-count metadata and control_features.log_author_count."
        ),
        "t0_boundary": (
            "Available no later than publication at T0. Derive only as "
            "expm1(log_author_count) on the audited overlap; missing or "
            "out-of-overlap records remain missing, never zero."
        ),
        "bias_risk": (
            "Field-specific team-size norms, consortium/group-author and partial "
            "byline handling, metadata coverage, and confounding with resources, "
            "prestige, or institutional capacity."
        ),
        "decision": "include",
        "reason": (
            "All five latest formalization mentions classify this as a publication-time "
            "context control. The table's source formula is preserved by the "
            "outcome-blind audit: expm1(log_author_count) equals the trusted author "
            "count for all 411489 audited overlap rows (rate 1.0)."
        ),
    },
    "reference_count": {
        "dimension_label": "Bibliographic knowledge-base structure",
        "dimension_definition": (
            "The count of backward references in a focal paper at publication, "
            "representing the size of its cited knowledge base as a contextual control "
            "rather than an innovation or impact outcome."
        ),
        "construct_role": "context_control",
        "information_source": (
            "Focal-paper backward-reference edges, represented by audited "
            "paper_references data and control_features.log_reference_count."
        ),
        "t0_boundary": (
            "Available no later than publication at T0. Derive only as "
            "expm1(log_reference_count) on audited covered rows; no observed backward "
            "edges is missing/unknown, never zero."
        ),
        "bias_risk": (
            "Field- and article-type-specific reference norms, incomplete citation-edge "
            "coverage, and confounding with manuscript length and knowledge-base breadth."
        ),
        "decision": "include",
        "reason": (
            "All four latest formalization mentions classify this as a publication-time "
            "context control. The table's source formula is preserved by the "
            "outcome-blind audit: expm1(log_reference_count) equals the backward-edge "
            "count across 354485 rows (rate 1.0), with audited coverage "
            "0.8614668643223408 and a fail-closed missing rule."
        ),
    },
}


def main() -> None:
    """Validate current evidence and populate only H1-editable coding fields."""
    columns, input_rows = read_csv(INPUT_PATH)
    if len(input_rows) != 2:
        raise ValueError("Expected exactly two dimension families")
    if not set(EDITABLE_FIELDS).issubset(columns):
        raise ValueError("Dimension-coding input lacks an editable H1 field")
    if {row["canonical_name_en"] for row in input_rows} != set(CODINGS):
        raise ValueError("Dimension families do not match the coding registry")
    if any(row["coder_role"] != "H1" for row in input_rows):
        raise ValueError("Round-2 input contains a non-H1 coding row")

    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if audit["schema_version"] != "operational_equivalence_audit_v4":
        raise ValueError("Unexpected operational-equivalence audit schema")
    if audit["outcome_columns_used"]:
        raise ValueError("Dimension coding cannot use outcome-derived evidence")
    for name in ("author_count", "reference_count"):
        if audit[name]["exact_equality_rate"] != 1.0:
            raise ValueError(f"{name} audit is not exact")

    completed_rows: list[dict[str, str]] = []
    for row in input_rows:
        completed = dict(row)
        completed.update(CODINGS[row["canonical_name_en"]])
        completed_rows.append(completed)
    write_csv(OUTPUT_PATH, columns, completed_rows)

    counts = {
        decision: sum(row["decision"] == decision for row in completed_rows)
        for decision in ("include", "exclude")
    }
    manifest = {
        "artifact": "dimension_coding_H1_round2_completed_v4",
        "reviewer": "H1",
        "input_path": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "operational_equivalence_audit_path": str(AUDIT_PATH),
        "operational_equivalence_audit_sha256": sha256(AUDIT_PATH),
        "output_path": str(OUTPUT_PATH),
        "output_sha256": sha256(OUTPUT_PATH),
        "family_rows": len(completed_rows),
        "editable_fields": EDITABLE_FIELDS,
        "decision_counts": counts,
        "qwen_or_ollama_used": False,
        "blind_review_constraints": [
            "Used only round-2 table-embedded evidence and the operational-equivalence audit.",
            "Did not use AI/H2 new output, a predictive model, Qwen, or Ollama.",
            "Filled only the eight H1-editable dimension fields and preserved all other input fields verbatim.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
