"""Import the three-role v3 coverage triage as a non-authorizing v4 ledger."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common import DATABASE_PATH, sha256_file, utc_now, write_json
from database import initialize, log_event

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "v3_coverage_scope_triage_H2_completed_v4.csv"
SUMMARY = ROOT / "outputs" / "v3_coverage_scope_triage_import_summary_v4.json"
DECISIONS = {"recover_priority", "scope_exclude", "needs_source_evidence"}
ROLES = {
    "direct_innovation",
    "t0_substantive",
    "t0_opportunity",
    "context_control",
    "out_of_scope",
    "uncertain",
}
REVIEWER_COLUMNS = {
    "AI": (
        "ai_triage_decision",
        "ai_scope_role_assessment",
        "ai_rationale",
        "ai_minimum_source_evidence_needed",
        "ai_search_terms_en",
    ),
    "H1": (
        "h1_triage_decision",
        "h1_scope_role_assessment",
        "h1_rationale",
        "h1_minimum_source_evidence_needed",
        "h1_search_terms_en",
    ),
    "H2": (
        "h2_final_triage_decision",
        "h2_final_scope_role_assessment",
        "h2_final_rationale",
        "h2_final_minimum_source_evidence_needed",
        "h2_final_search_terms_en",
    ),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a completed H2 triage sheet."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 432:
        raise ValueError(f"Expected 432 H2 triage rows, found {len(rows)}")
    return rows


def validate(row: dict[str, str], reviewer: str) -> tuple[str, str, str, str, str]:
    """Validate one role's fixed-vocabulary decision payload."""
    columns = REVIEWER_COLUMNS[reviewer]
    values = tuple(str(row.get(column, "")).strip() for column in columns)
    decision, scope_role, rationale, minimum, terms = values
    if decision not in DECISIONS:
        raise ValueError(f"{reviewer} invalid decision for {row.get('v3_feature_id')}")
    if scope_role not in ROLES:
        raise ValueError(
            f"{reviewer} invalid scope role for {row.get('v3_feature_id')}"
        )
    if not rationale:
        raise ValueError(f"{reviewer} blank rationale for {row.get('v3_feature_id')}")
    if decision != "scope_exclude" and not terms:
        raise ValueError(
            f"{reviewer} needs English search terms for {row.get('v3_feature_id')}"
        )
    return decision, scope_role, rationale, minimum, terms


def import_triage(connection: Any, path: Path, summary_path: Path) -> dict[str, Any]:
    """Store all independent and adjudicated decisions without selecting features."""
    rows = read_rows(path)
    digest = sha256_file(path)
    counts: Counter[str] = Counter()
    required_count = 0
    seen: set[str] = set()
    for row in rows:
        feature_id = str(row.get("v3_feature_id", "")).strip()
        if not feature_id or feature_id in seen:
            raise ValueError(f"Duplicate/missing v3 feature ID: {feature_id!r}")
        seen.add(feature_id)
        canonical = str(row.get("canonical_name_en", "")).strip()
        if not canonical:
            raise ValueError(f"Missing canonical label for {feature_id}")
        existing = connection.execute(
            "SELECT v3_canonical_name_en FROM v3_coverage_reconciliation WHERE v3_feature_id = ?",
            (feature_id,),
        ).fetchone()
        if existing is None or str(existing[0]) != canonical:
            raise ValueError(f"Unregistered/mismatched coverage label: {feature_id}")
        reviews = {role: validate(row, role) for role in REVIEWER_COLUMNS}
        h2 = reviews["H2"]
        counts[h2[0]] += 1
        required_count += int(str(row.get("h2_review_required", "")) == "1")
        for role, payload in reviews.items():
            connection.execute(
                """
                INSERT INTO v3_coverage_triage_reviews(
                    v3_feature_id, reviewer_role, triage_decision,
                    scope_role_assessment, rationale,
                    minimum_source_evidence_needed, search_terms_en,
                    artifact_path, artifact_sha256, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(v3_feature_id, reviewer_role) DO UPDATE SET
                    triage_decision = excluded.triage_decision,
                    scope_role_assessment = excluded.scope_role_assessment,
                    rationale = excluded.rationale,
                    minimum_source_evidence_needed = excluded.minimum_source_evidence_needed,
                    search_terms_en = excluded.search_terms_en,
                    artifact_path = excluded.artifact_path,
                    artifact_sha256 = excluded.artifact_sha256,
                    imported_at = excluded.imported_at
                """,
                (feature_id, role, *payload, str(path.resolve()), digest, utc_now()),
            )
        disposition = {
            "recover_priority": "source_recovery_approved_h2",
            "scope_exclude": "scope_excluded_h2",
            "needs_source_evidence": "source_evidence_required_h2",
        }[h2[0]]
        connection.execute(
            """
            UPDATE v3_coverage_reconciliation
            SET coverage_disposition = ?, evidence_status = 'triage_adjudicated',
                final_reason = ?, reviewed_by = 'H2', reviewed_at = ?
            WHERE v3_feature_id = ?
            """,
            (disposition, h2[2], utc_now(), feature_id),
        )
    result = {
        "schema_version": "v3_coverage_scope_triage_import_v4",
        "artifact_path": str(path.resolve()),
        "artifact_sha256": digest,
        "row_count": len(rows),
        "h2_review_required_count": required_count,
        "h2_counts": dict(sorted(counts.items())),
        "selection_authorization": False,
        "interpretation": "H2 triage decides source-recovery routing only; it does not establish formula evidence, data mapping, dimensions, or final feature inclusion.",
    }
    log_event(
        connection, "v3_coverage_triage_import", "coverage_anchor", "v3_432", result
    )
    connection.commit()
    write_json(summary_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        print(
            json.dumps(
                import_triage(connection, args.input.resolve(), args.summary.resolve()),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
