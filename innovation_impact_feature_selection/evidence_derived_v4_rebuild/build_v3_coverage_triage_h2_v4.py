"""Build an auditable H2 adjudication sheet for blind v3-coverage triage."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from common import (
    cohen_kappa,
    gwet_ac1,
    raw_agreement,
    sha256_file,
    write_csv,
    write_json,
)

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "v3_coverage_scope_triage_input_v4.csv"
AI = ROOT / "outputs" / "v3_coverage_scope_triage_AI_completed_v4.csv"
H1 = ROOT / "outputs" / "v3_coverage_scope_triage_H1_completed_v4.csv"
OUTPUT = ROOT / "outputs" / "v3_coverage_scope_triage_H2_v4.csv"
MANIFEST = ROOT / "outputs" / "v3_coverage_scope_triage_H2_v4.manifest.json"
DECISIONS = {"recover_priority", "scope_exclude", "needs_source_evidence"}
ROLES = {
    "direct_innovation",
    "t0_substantive",
    "t0_opportunity",
    "context_control",
    "out_of_scope",
    "uncertain",
}
BASE_FIELDS = (
    "v3_feature_id",
    "canonical_name_en",
    "archived_scope_role",
    "archived_t0_claim",
    "archived_formula_text",
    "archived_english_fulltext_verified",
)
REVIEW_FIELDS = (
    "triage_decision",
    "scope_role_assessment",
    "rationale",
    "minimum_source_evidence_needed",
    "search_terms_en",
)
OUTPUT_FIELDS = (
    *BASE_FIELDS,
    *(f"ai_{field}" for field in REVIEW_FIELDS),
    *(f"h1_{field}" for field in REVIEW_FIELDS),
    "h2_review_required",
    "h2_final_triage_decision",
    "h2_final_scope_role_assessment",
    "h2_final_rationale",
    "h2_final_minimum_source_evidence_needed",
    "h2_final_search_terms_en",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one triage sheet with a stable UTF-8 CSV dialect."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_completed(
    rows: list[dict[str, str]], role: str
) -> dict[str, dict[str, str]]:
    """Validate every independent reviewer completed all allowed decisions."""
    if len(rows) != 432:
        raise ValueError(f"{role} must contain 432 rows, found {len(rows)}")
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        feature_id = row.get("v3_feature_id", "")
        if not feature_id or feature_id in by_id:
            raise ValueError(f"{role} has duplicate/missing feature ID: {feature_id!r}")
        if row.get("triage_decision") not in DECISIONS:
            raise ValueError(f"{role} has invalid triage decision for {feature_id}")
        if row.get("scope_role_assessment") not in ROLES:
            raise ValueError(f"{role} has invalid scope role for {feature_id}")
        if not row.get("rationale", "").strip():
            raise ValueError(f"{role} has no rationale for {feature_id}")
        by_id[feature_id] = row
    return by_id


def build(
    input_path: Path, ai_path: Path, h1_path: Path, output: Path, manifest: Path
) -> dict[str, Any]:
    """Join independent reviews without silently choosing either reviewer's view."""
    source_rows = read_csv(input_path)
    ai = validate_completed(read_csv(ai_path), "AI")
    h1 = validate_completed(read_csv(h1_path), "H1")
    if len(source_rows) != 432:
        raise ValueError(f"Input must contain 432 rows, found {len(source_rows)}")
    output_rows: list[dict[str, str]] = []
    ai_decisions: list[str] = []
    h1_decisions: list[str] = []
    review_required = 0
    for source in source_rows:
        feature_id = source["v3_feature_id"]
        if feature_id not in ai or feature_id not in h1:
            raise ValueError(f"Missing independent review for {feature_id}")
        ai_row = ai[feature_id]
        h1_row = h1[feature_id]
        ai_decisions.append(ai_row["triage_decision"])
        h1_decisions.append(h1_row["triage_decision"])
        required = (
            ai_row["triage_decision"] != h1_row["triage_decision"]
            or ai_row["scope_role_assessment"] != h1_row["scope_role_assessment"]
            or ai_row["triage_decision"] != "scope_exclude"
            or h1_row["triage_decision"] != "scope_exclude"
        )
        review_required += int(required)
        row = {field: source.get(field, "") for field in BASE_FIELDS}
        row.update({f"ai_{field}": ai_row.get(field, "") for field in REVIEW_FIELDS})
        row.update({f"h1_{field}": h1_row.get(field, "") for field in REVIEW_FIELDS})
        row.update(
            {
                "h2_review_required": "1" if required else "0",
                "h2_final_triage_decision": "",
                "h2_final_scope_role_assessment": "",
                "h2_final_rationale": "",
                "h2_final_minimum_source_evidence_needed": "",
                "h2_final_search_terms_en": "",
            }
        )
        output_rows.append(row)
    write_csv(output, output_rows, OUTPUT_FIELDS)
    result = {
        "schema_version": "v3_coverage_scope_triage_h2_v4",
        "input_sha256": sha256_file(input_path),
        "ai_sha256": sha256_file(ai_path),
        "h1_sha256": sha256_file(h1_path),
        "output_path": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "row_count": len(output_rows),
        "h2_review_required_count": review_required,
        "agreement": {
            "raw_agreement": raw_agreement(ai_decisions, h1_decisions),
            "cohen_kappa": cohen_kappa(ai_decisions, h1_decisions),
            "gwet_ac1": gwet_ac1(ai_decisions, h1_decisions),
            "ai_counts": dict(sorted(Counter(ai_decisions).items())),
            "h1_counts": dict(sorted(Counter(h1_decisions).items())),
        },
    }
    write_json(manifest, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--ai", type=Path, default=AI)
    parser.add_argument("--h1", type=Path, default=H1)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    print(
        build(
            args.input.resolve(),
            args.ai.resolve(),
            args.h1.resolve(),
            args.output.resolve(),
            args.manifest.resolve(),
        )
    )


if __name__ == "__main__":
    main()
