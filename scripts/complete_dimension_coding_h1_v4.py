"""Complete the independent H1 dimension coding for audited count families."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Final

ROOT: Final = Path("/home/jayee/workspace/ASPR")
INPUT_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "dimension_coding_H1_v4.csv"
)
FORMALIZATION_PATH: Final = (
    ROOT / "outputs/contextual_formalization_H1_batch3_completed_v4.csv"
)
AUDIT_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "operational_equivalence_audit_v4.json"
)
OUTPUT_PATH: Final = ROOT / "outputs/dimension_coding_H1_completed_v4.csv"
MANIFEST_PATH: Final = ROOT / "outputs/dimension_coding_H1_completed_v4.manifest.json"
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
    """Read a UTF-8 CSV while preserving its header order."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    """Write a CSV without changing its schema."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def verify_frozen_formalization() -> None:
    """Confirm the H1 formalization decisions relied on for this coding."""
    _, rows = read_csv(FORMALIZATION_PATH)
    decisions = {
        row["H1_canonical_name_en"]: row["H1_formalization_decision"]
        for row in rows
        if row["H1_canonical_name_en"] in {"author_count", "reference_count"}
    }
    expected = {
        "author_count": "promote_for_formalization",
        "reference_count": "promote_for_formalization",
    }
    if decisions != expected:
        raise ValueError("Frozen H1 formalization does not support both count families")


CODINGS: Final[dict[str, dict[str, str]]] = {
    "author_count": {
        "dimension_label": "Team composition and collaboration scale",
        "dimension_definition": (
            "The number of listed authors on a focal paper at publication, used to "
            "represent the scale of the producing team as a contextual control rather "
            "than an innovation or impact outcome."
        ),
        "construct_role": "context_control",
        "information_source": (
            "Focal-paper author byline, represented by the audited T0 "
            "control_features.log_author_count and OpenAlex author-count metadata."
        ),
        "t0_boundary": (
            "Available no later than publication at T0. Use expm1(log_author_count) "
            "only in the audited overlap; absent or out-of-overlap values remain "
            "missing and are never imputed or recoded to zero."
        ),
        "bias_risk": (
            "Field-specific team-size norms, group/consortium and incomplete-byline "
            "handling, author-metadata coverage, and confounding with resources, "
            "prestige, or institutional capacity."
        ),
        "decision": "include",
        "reason": (
            "The table identifies a publication-time context control. Frozen H1 "
            "formalization and the operational-equivalence audit establish an exact, "
            "outcome-blind representation transform (411489 overlap rows; equality "
            "rate 1.0), so this family is included solely as a team-scale control."
        ),
    },
    "reference_count": {
        "dimension_label": "Bibliographic knowledge-base structure",
        "dimension_definition": (
            "The size of the focal paper's backward reference list at publication, "
            "representing the scale of its cited knowledge base as a contextual "
            "control rather than an innovation or impact outcome."
        ),
        "construct_role": "context_control",
        "information_source": (
            "Focal-paper backward reference edges, represented by audited T0 "
            "control_features.log_reference_count and paper_references edge data."
        ),
        "t0_boundary": (
            "Available no later than publication at T0. Use "
            "expm1(log_reference_count) only for audited covered rows; no observed "
            "backward edges means missing/unknown, never zero."
        ),
        "bias_risk": (
            "Field- and article-type-specific referencing norms, incomplete "
            "citation-edge coverage, and confounding with manuscript length and "
            "knowledge-base breadth."
        ),
        "decision": "include",
        "reason": (
            "The table identifies a publication-time context control. Frozen H1 "
            "formalization and the audit establish an exact, outcome-blind inverse "
            "transform over 354485 rows (equality rate 1.0), with 0.8614668643223408 "
            "coverage and an explicit fail-closed missing rule."
        ),
    },
}


def main() -> None:
    """Validate frozen evidence and write only the editable coding fields."""
    columns, input_rows = read_csv(INPUT_PATH)
    if len(input_rows) != 2:
        raise ValueError("Expected exactly two dimension families")
    if not set(EDITABLE_FIELDS).issubset(columns):
        raise ValueError("Dimension-coding input lacks an editable H1 field")
    if {row["canonical_name_en"] for row in input_rows} != set(CODINGS):
        raise ValueError("Dimension families do not match the coding registry")

    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if audit["schema_version"] != "operational_equivalence_audit_v4":
        raise ValueError("Unexpected operational-equivalence audit schema")
    if audit["outcome_columns_used"]:
        raise ValueError("Dimension coding cannot use outcome-derived evidence")
    for count_name in ("author_count", "reference_count"):
        if audit[count_name]["exact_equality_rate"] != 1.0:
            raise ValueError(f"{count_name} audit is not exact")
    verify_frozen_formalization()

    completed_rows: list[dict[str, str]] = []
    for row in input_rows:
        completed = dict(row)
        completed.update(CODINGS[row["canonical_name_en"]])
        completed_rows.append(completed)
    write_csv(OUTPUT_PATH, columns, completed_rows)

    decisions = {
        decision: sum(row["decision"] == decision for row in completed_rows)
        for decision in ("include", "exclude")
    }
    manifest = {
        "artifact": "dimension_coding_H1_completed_v4",
        "reviewer": "H1",
        "input_path": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "frozen_formalization_path": str(FORMALIZATION_PATH),
        "frozen_formalization_sha256": sha256(FORMALIZATION_PATH),
        "operational_equivalence_audit_path": str(AUDIT_PATH),
        "operational_equivalence_audit_sha256": sha256(AUDIT_PATH),
        "output_path": str(OUTPUT_PATH),
        "output_sha256": sha256(OUTPUT_PATH),
        "family_rows": len(completed_rows),
        "editable_fields": EDITABLE_FIELDS,
        "decision_counts": decisions,
        "qwen_or_ollama_used": False,
        "blind_review_constraints": [
            "Used only table-embedded evidence, frozen H1 formalization, and the operational-equivalence audit.",
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
