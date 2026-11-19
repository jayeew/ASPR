"""Create blind scope-triage materials for the recovered v3 coverage anchor.

This is deliberately earlier than source/formula extraction.  The archived
432 labels include broad discovery noise; treating every label as a v4
candidate would reproduce that noise, while ignoring them would repeat the
coverage failure.  Independent reviewers therefore classify only the next
evidence-recovery action.  No triage decision is a feature decision.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from common import sha256_file, write_csv, write_json

ROOT = Path(__file__).resolve().parent
ANCHOR = ROOT / "outputs" / "v3_coverage_anchor" / "complete_indicator_library_v3.csv"
OUTPUT = ROOT / "outputs" / "v3_coverage_scope_triage_input_v4.csv"
MANIFEST = ROOT / "outputs" / "v3_coverage_scope_triage_input_v4.manifest.json"
BRIEF = ROOT / "V3_COVERAGE_SCOPE_TRIAGE_BRIEF_V4.md"
FIELDS = (
    "v3_feature_id",
    "canonical_name_en",
    "archived_scope_role",
    "archived_t0_claim",
    "archived_formula_text",
    "archived_english_fulltext_verified",
    "triage_decision",
    "scope_role_assessment",
    "rationale",
    "minimum_source_evidence_needed",
    "search_terms_en",
)


def read_rows(anchor: Path) -> list[dict[str, str]]:
    """Read exactly the immutable archived coverage labels."""
    with anchor.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 432:
        raise ValueError(f"Expected 432 archived labels, found {len(rows)}")
    return rows


def build(anchor: Path, output: Path, manifest: Path, brief: Path) -> dict[str, Any]:
    """Write a blank, reviewer-neutral scope triage sheet and instructions."""
    rows = read_rows(anchor)
    output_rows = [
        {
            "v3_feature_id": row["feature_id"],
            "canonical_name_en": row["canonical_name_en"],
            "archived_scope_role": row["scope_role"],
            "archived_t0_claim": row["maximum_information_time"],
            "archived_formula_text": row["formula_text"],
            "archived_english_fulltext_verified": row["english_fulltext_verified"],
            "triage_decision": "",
            "scope_role_assessment": "",
            "rationale": "",
            "minimum_source_evidence_needed": "",
            "search_terms_en": "",
        }
        for row in rows
    ]
    write_csv(output, output_rows, FIELDS)
    brief.write_text(
        """# Blind v3-coverage scope triage for v4

## Purpose

The attached 432 labels are a **historical coverage benchmark**, not
pre-approved v4 features.  Independently decide only the next recovery action
for each label.  Do not use model performance, the legacy dimension label, or
another reviewer's file.

## Allowed `triage_decision` values

- `recover_priority`: plausibly a paper-level feature of innovation, T0
  substantive potential, T0 opportunity, or a paper-level background control;
  recover original English source evidence.
- `scope_exclude`: clearly a clinical/study-specific outcome, systematic-review
  procedure, non-paper-level construct, post-publication result, or other
  construct outside the v4 target.  State the fixed reason.
- `needs_source_evidence`: label alone is ambiguous; obtain source evidence
  before a scope judgment.

## Required columns

- `scope_role_assessment`: one of `direct_innovation`, `t0_substantive`,
  `t0_opportunity`, `context_control`, `out_of_scope`, or `uncertain`.
- `rationale`: concise construct-level reason.  A legacy role/T0 claim is not
  evidence.
- `minimum_source_evidence_needed`: original application, mathematical
  foundation, validation, or `none_for_clear_scope_exclusion`.
- `search_terms_en`: 2–6 source-search English terms/phrases.  Leave blank
  only for a clear scope exclusion.

## Prohibitions

- This is not formula verification, data mapping, dimension formation, or
  final feature selection.
- Do not approve an item merely because it is in the 432-label archive.
- Do not exclude an ambiguous item solely because its archived formula is
  absent; choose `needs_source_evidence` instead.
""",
        encoding="utf-8",
    )
    result = {
        "schema_version": "v3_coverage_scope_triage_v4",
        "input_path": str(output.resolve()),
        "input_sha256": sha256_file(output),
        "brief_path": str(brief.resolve()),
        "brief_sha256": sha256_file(brief),
        "anchor_path": str(anchor.resolve()),
        "anchor_sha256": sha256_file(anchor),
        "row_count": len(rows),
        "purpose": "blind_next_action_triage_not_feature_selection",
    }
    write_json(manifest, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, default=ANCHOR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--brief", type=Path, default=BRIEF)
    args = parser.parse_args()
    print(
        build(
            args.anchor.resolve(),
            args.output.resolve(),
            args.manifest.resolve(),
            args.brief.resolve(),
        )
    )


if __name__ == "__main__":
    main()
