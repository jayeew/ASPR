"""Import the three-role contextual source screen without altering formal screening."""

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
INPUT = ROOT / "outputs" / "contextual_source_screening_H2_completed_v4.csv"
SUMMARY = ROOT / "outputs" / "contextual_source_screening_import_summary_v4.json"
DECISIONS = {"include_definition_or_review", "exclude_not_relevant", "uncertain"}
ROLE_COLUMNS = {
    "AI": ("ai_screen_decision", "ai_evidence_span", "ai_rationale"),
    "H1": ("h1_screen_decision", "h1_evidence_span", "h1_rationale"),
    "H2": (
        "h2_final_screen_decision",
        "h2_final_evidence_span",
        "h2_final_rationale",
    ),
}


def read_rows(path: Path, expected_rows: int | None) -> list[dict[str, str]]:
    """Read the completed, fully adjudicated source-screening sheet."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} source-screen rows, found {len(rows)}"
        )
    return rows


def payload(row: dict[str, str], role: str) -> tuple[str, str, str]:
    """Validate one role's source-lead disposition."""
    decision, span, rationale = (
        str(row.get(value, "")).strip() for value in ROLE_COLUMNS[role]
    )
    if decision not in DECISIONS:
        raise ValueError(f"{role} invalid decision for {row.get('record_key')}")
    if not rationale:
        raise ValueError(f"{role} blank rationale for {row.get('record_key')}")
    if decision != "exclude_not_relevant" and not span:
        # The blind batch-7 sheets deliberately expose only decision/rationale
        # fields.  Preserve their reason as the audit span rather than reject a
        # valid independent review solely for that schema difference.  H2's
        # final sheet remains required to contain a direct evidence span.
        if role == "H2":
            raise ValueError(f"{role} blank evidence span for {row.get('record_key')}")
        span = rationale
    return decision, span, rationale


def import_sheet(
    connection: Any,
    path: Path,
    summary: Path,
    expected_rows: int | None,
    batch_id: str,
) -> dict[str, Any]:
    """Persist every role's decision and H2's final source-recovery route."""
    rows = read_rows(path, expected_rows)
    digest = sha256_file(path)
    counts: Counter[str] = Counter()
    required = 0
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("record_key", "")).strip()
        if not key or key in seen:
            raise ValueError(f"Duplicate/missing record key: {key!r}")
        seen.add(key)
        if (
            connection.execute(
                "SELECT 1 FROM records WHERE record_key = ?", (key,)
            ).fetchone()
            is None
        ):
            raise ValueError(f"Contextual screen record is absent: {key}")
        values = {role: payload(row, role) for role in ROLE_COLUMNS}
        h2 = values["H2"]
        counts[h2[0]] += 1
        h2_required = int(str(row.get("h2_review_required", "")) == "1")
        required += h2_required
        for role, decision in values.items():
            connection.execute(
                """
                INSERT INTO contextual_source_screening_reviews(
                    record_key, reviewer_role, screen_decision, evidence_span,
                    rationale, artifact_path, artifact_sha256, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(record_key, reviewer_role) DO UPDATE SET
                    screen_decision = excluded.screen_decision,
                    evidence_span = excluded.evidence_span,
                    rationale = excluded.rationale,
                    artifact_path = excluded.artifact_path,
                    artifact_sha256 = excluded.artifact_sha256,
                    imported_at = excluded.imported_at
                """,
                (key, role, *decision, str(path.resolve()), digest, utc_now()),
            )
        connection.execute(
            """
            INSERT INTO contextual_source_final(
                record_key, final_decision, evidence_span, rationale,
                h2_required, h2_completed, finalized_at
            ) VALUES (?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(record_key) DO UPDATE SET
                final_decision = excluded.final_decision,
                evidence_span = excluded.evidence_span,
                rationale = excluded.rationale,
                h2_required = excluded.h2_required,
                h2_completed = excluded.h2_completed,
                finalized_at = excluded.finalized_at
            """,
            (key, *h2, h2_required, utc_now()),
        )
    result = {
        "schema_version": "contextual_source_screening_import_v4",
        "artifact_path": str(path.resolve()),
        "artifact_sha256": digest,
        "row_count": len(rows),
        "h2_review_required_count": required,
        "h2_counts": dict(sorted(counts.items())),
        "formal_literature_screening_changed": False,
        "selection_authorization": False,
    }
    log_event(
        connection,
        "contextual_source_screening_import",
        "contextual_probe",
        batch_id,
        result,
    )
    connection.commit()
    write_json(summary, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--expected-rows", type=int, default=120)
    parser.add_argument("--batch-id", default="batch_001")
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        print(
            json.dumps(
                import_sheet(
                    connection,
                    args.input.resolve(),
                    args.summary.resolve(),
                    args.expected_rows,
                    str(args.batch_id),
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
