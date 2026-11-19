"""Validate and persist H2-adjudicated pre-promotion candidate coding."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from common import DATABASE_PATH, parse_bool, sha256_file, utc_now, write_json
from database import initialize, log_event

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
INPUT = OUTPUT / "contextual_candidate_canonicalization_H2_v4.csv"
COMPLETED = OUTPUT / "contextual_candidate_canonicalization_H2_completed_v4.csv"
SUMMARY = OUTPUT / "contextual_candidate_canonicalization_H2_import_summary_v4.json"
FIELDS = (
    "family_name_en",
    "merge_or_split_reason",
    "formula_reproducible",
    "t0_computable",
    "scope_role",
    "missing_rule_status",
    "promotion_decision",
    "rationale",
)
SCOPE_ROLES = {
    "direct_innovation",
    "t0_substantive",
    "t0_opportunity",
    "context_control",
    "out_of_scope",
}
PROMOTION_DECISIONS = {
    "promote_for_formalization",
    "retain_evidence_gap",
    "reject",
}
PROMOTION_DECISION_ALIASES = {
    "hold_pending_specification": "retain_evidence_gap",
}
MISSING_RULE_STATUSES = {"explicit", "derivable_from_source", "absent"}
MISSING_RULE_ALIASES = {
    "stated": "explicit",
    "needs_formalization": "derivable_from_source",
    "source_rule_insufficient": "absent",
    "source_rule_needs_formalization": "absent",
    "source_condition_present_rule_needs_formalization": "derivable_from_source",
}


def _read(path: Path) -> list[dict[str, str]]:
    """Read a CSV with its header preserved as mapping keys."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader)


def _index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Index rows by immutable candidate ID."""
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        key = str(row.get("candidate_id", "")).strip()
        if not key or key in indexed:
            raise ValueError(f"Blank/duplicate candidate ID: {key!r}")
        indexed[key] = row
    return indexed


def _validate_protected(
    before: list[dict[str, str]], after: list[dict[str, str]]
) -> None:
    """Reject an H2 sheet that edited blind-review or source evidence fields."""
    original, completed = _index(before), _index(after)
    if set(original) != set(completed):
        raise ValueError("Completed H2 sheet changes the frozen candidate set.")
    allowed = {f"h2_{field}" for field in FIELDS}
    for key, source in original.items():
        changed = {
            field
            for field in set(source) | set(completed[key])
            if str(source.get(field, "")) != str(completed[key].get(field, ""))
        }
        if not changed <= allowed:
            raise ValueError(
                f"H2 changed protected fields for {key}: {sorted(changed)}"
            )


def _payload(row: dict[str, str], prefix: str) -> tuple[str, ...]:
    """Validate one review payload and normalize its Boolean fields."""
    values = tuple(str(row.get(f"{prefix}_{field}", "")).strip() for field in FIELDS)
    if any(not value for value in values):
        raise ValueError(f"{prefix} has a blank canonicalization field")
    family, merge, formula, t0, scope, missing, decision, rationale = values
    missing = MISSING_RULE_ALIASES.get(missing, missing)
    if missing not in MISSING_RULE_STATUSES and missing.startswith("source_"):
        missing = (
            "derivable_from_source" if "condition_present" in missing else "absent"
        )
    if missing not in MISSING_RULE_STATUSES and missing.startswith("external_"):
        missing = "absent"
    decision = PROMOTION_DECISION_ALIASES.get(decision, decision)
    if scope not in SCOPE_ROLES or decision not in PROMOTION_DECISIONS:
        raise ValueError(f"Invalid {prefix} role or promotion decision")
    if missing not in MISSING_RULE_STATUSES:
        raise ValueError(f"Invalid {prefix} missing-rule status")
    return (
        family,
        merge,
        str(int(parse_bool(formula, f"{prefix}_formula_reproducible"))),
        str(int(parse_bool(t0, f"{prefix}_t0_computable"))),
        scope,
        missing,
        decision,
        rationale,
    )


def import_sheet(
    connection: sqlite3.Connection, input_path: Path, completed_path: Path
) -> dict[str, Any]:
    """Persist the three role-specific decisions, without feature selection."""
    before, after = _read(input_path), _read(completed_path)
    _validate_protected(before, after)
    digest = sha256_file(completed_path)
    counts: dict[str, int] = {}
    for row in after:
        candidate_id = str(row["candidate_id"]).strip()
        for role, prefix in (("AI", "ai"), ("H1", "h1"), ("H2", "h2")):
            family, merge, formula, t0, scope, missing, decision, rationale = _payload(
                row, prefix
            )
            connection.execute(
                """
                INSERT INTO contextual_candidate_canonicalization_reviews(
                    candidate_id, reviewer_role, family_name_en, merge_or_split_reason,
                    formula_reproducible, t0_computable, scope_role,
                    missing_rule_status, promotion_decision, rationale,
                    artifact_path, artifact_sha256, reviewed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id, reviewer_role) DO UPDATE SET
                    family_name_en=excluded.family_name_en,
                    merge_or_split_reason=excluded.merge_or_split_reason,
                    formula_reproducible=excluded.formula_reproducible,
                    t0_computable=excluded.t0_computable, scope_role=excluded.scope_role,
                    missing_rule_status=excluded.missing_rule_status,
                    promotion_decision=excluded.promotion_decision,
                    rationale=excluded.rationale, artifact_path=excluded.artifact_path,
                    artifact_sha256=excluded.artifact_sha256, reviewed_at=excluded.reviewed_at
                """,
                (
                    candidate_id,
                    role,
                    family,
                    merge,
                    int(formula),
                    int(t0),
                    scope,
                    missing,
                    decision,
                    rationale,
                    str(completed_path.resolve()),
                    digest,
                    utc_now(),
                ),
            )
            if role == "H2":
                counts[decision] = counts.get(decision, 0) + 1
                connection.execute(
                    """
                    INSERT INTO contextual_candidate_canonicalization_final(
                        candidate_id, family_name_en, merge_or_split_reason,
                        formula_reproducible, t0_computable, scope_role,
                        missing_rule_status, promotion_decision, rationale,
                        artifact_path, artifact_sha256, finalized_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(candidate_id) DO UPDATE SET
                        family_name_en=excluded.family_name_en,
                        merge_or_split_reason=excluded.merge_or_split_reason,
                        formula_reproducible=excluded.formula_reproducible,
                        t0_computable=excluded.t0_computable, scope_role=excluded.scope_role,
                        missing_rule_status=excluded.missing_rule_status,
                        promotion_decision=excluded.promotion_decision,
                        rationale=excluded.rationale, artifact_path=excluded.artifact_path,
                        artifact_sha256=excluded.artifact_sha256, finalized_at=excluded.finalized_at
                    """,
                    (
                        candidate_id,
                        family,
                        merge,
                        int(formula),
                        int(t0),
                        scope,
                        missing,
                        decision,
                        rationale,
                        str(completed_path.resolve()),
                        digest,
                        utc_now(),
                    ),
                )
    result = {
        "schema_version": "contextual_candidate_canonicalization_import_v4",
        "candidate_count": len(after),
        "h2_promotion_counts": counts,
        "artifact_path": str(completed_path.resolve()),
        "artifact_sha256": digest,
        "selection_authorization": False,
    }
    log_event(
        connection,
        "contextual_candidate_canonicalization_import",
        "candidate_canonicalization",
        "h2_pre_promotion",
        result,
    )
    connection.commit()
    return result


def main() -> None:
    """Load the H2 sheet and write its auditable database state."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--completed", type=Path, default=COMPLETED)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        result = import_sheet(
            connection, args.input.resolve(), args.completed.resolve()
        )
    finally:
        connection.close()
    write_json(args.summary.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
