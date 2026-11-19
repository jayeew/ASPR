"""Validate and persist H2 formula/data formalization decisions."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from build_contextual_formalization_h2_v4 import FIELDS
from common import DATABASE_PATH, sha256_file, utc_now, write_json
from database import initialize, log_event

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
INPUT = OUTPUT / "contextual_formalization_H2_v4.csv"
COMPLETED = OUTPUT / "contextual_formalization_H2_completed_v4.csv"
SUMMARY = OUTPUT / "contextual_formalization_H2_import_summary_v4.json"
DECISIONS = {"promote_for_formalization", "retain_evidence_gap", "reject"}


def _read(path: Path) -> list[dict[str, str]]:
    """Read the CSV as UTF-8 dictionaries."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader)


def _index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Index one sheet by immutable candidate ID."""
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        key = str(row.get("candidate_id", "")).strip()
        if not key or key in result:
            raise ValueError(f"Blank/duplicate candidate ID: {key!r}")
        result[key] = row
    return result


def _protected(before: list[dict[str, str]], after: list[dict[str, str]]) -> None:
    """Require H2 to alter only its explicitly editable fields."""
    original, completed = _index(before), _index(after)
    if set(original) != set(completed):
        raise ValueError("H2 output changes the candidate identity set")
    allowed = {f"h2_{field}" for field in FIELDS}
    for key, row in original.items():
        changed = {
            field
            for field in set(row) | set(completed[key])
            if str(row.get(field, "")) != str(completed[key].get(field, ""))
        }
        if not changed <= allowed:
            raise ValueError(
                f"H2 changed protected fields for {key}: {sorted(changed)}"
            )


def _payload(row: dict[str, str], prefix: str) -> dict[str, str]:
    """Read a nonblank role payload and validate its structured JSON cells."""
    values = {field: str(row.get(f"{prefix}_{field}", "")).strip() for field in FIELDS}
    if any(not value for value in values.values()):
        raise ValueError(f"{prefix} has a blank formalization field")
    if values["formalization_decision"] not in DECISIONS:
        raise ValueError(f"Invalid {prefix} formalization decision")
    for field in ("required_data_json", "local_source_ids_json", "local_columns_json"):
        if not isinstance(json.loads(values[field]), list):
            raise TypeError(f"{prefix} {field} must be a JSON list")
    return values


def import_sheet(
    connection: sqlite3.Connection, input_path: Path, completed_path: Path
) -> dict[str, Any]:
    """Persist all three reviews and the H2 final payload, without selection."""
    before, after = _read(input_path), _read(completed_path)
    _protected(before, after)
    digest = sha256_file(completed_path)
    counts: dict[str, int] = {}
    for row in after:
        candidate_id = str(row["candidate_id"])
        for role, prefix in (("AI", "ai"), ("H1", "h1"), ("H2", "h2")):
            payload = _payload(row, prefix)
            connection.execute(
                """
                INSERT INTO contextual_formalization_reviews(
                    candidate_id, reviewer_role, payload_json, artifact_path,
                    artifact_sha256, reviewed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id, reviewer_role) DO UPDATE SET
                    payload_json=excluded.payload_json, artifact_path=excluded.artifact_path,
                    artifact_sha256=excluded.artifact_sha256, reviewed_at=excluded.reviewed_at
                """,
                (
                    candidate_id,
                    role,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    str(completed_path.resolve()),
                    digest,
                    utc_now(),
                ),
            )
            if role != "H2":
                continue
            counts[payload["formalization_decision"]] = (
                counts.get(payload["formalization_decision"], 0) + 1
            )
            columns = (
                "candidate_id",
                *FIELDS,
                "artifact_path",
                "artifact_sha256",
                "finalized_at",
            )
            values = (
                candidate_id,
                *[payload[field] for field in FIELDS],
                str(completed_path.resolve()),
                digest,
                utc_now(),
            )
            assignments = ", ".join(
                f"{field}=excluded.{field}" for field in columns[1:]
            )
            connection.execute(
                f"INSERT INTO contextual_formalization_final({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)}) "
                f"ON CONFLICT(candidate_id) DO UPDATE SET {assignments}",
                values,
            )
    result = {
        "schema_version": "contextual_formalization_import_v4",
        "candidate_count": len(after),
        "h2_decision_counts": counts,
        "artifact_path": str(completed_path.resolve()),
        "artifact_sha256": digest,
        "selection_authorization": False,
    }
    log_event(
        connection, "contextual_formalization_import", "formalization", "h2", result
    )
    connection.commit()
    return result


def main() -> None:
    """Run the formalization importer."""
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
