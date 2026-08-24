"""Conservatively normalize legacy review-session provenance payloads."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    from .core import ProtocolError, canonical_json, file_hash
except ImportError:  # pragma: no cover - direct script execution
    from core import ProtocolError, canonical_json, file_hash  # type: ignore[no-redef]

PROTECTED_COLUMNS = (
    "review_session_id",
    "run_id",
    "reviewer_role",
    "input_hash",
    "output_hash",
    "model_label",
    "reason",
    "created_at",
)


def _existing_payload(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _artifact_path(value: str) -> Path:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    if not isinstance(parsed, str) or not parsed.strip():
        raise ProtocolError("legacy evidence is not an artifact path")
    path = Path(parsed).expanduser().resolve()
    if not path.is_file():
        raise ProtocolError(f"review artifact not found: {path}")
    if path.suffix.casefold() != ".csv":
        raise ProtocolError(f"review artifact is not CSV: {path}")
    return path


def _stage(reason: str, path: Path) -> str:
    reason_key = reason.casefold()
    name = path.name.casefold()
    candidates: list[str] = []
    if any(
        token in reason_key for token in ("terminology", "press", "query")
    ) and name.startswith(("search_frame_", "final_search_")):
        candidates.append("search-frame")
    if any(
        token in reason_key
        for token in ("formal literature screening", "dual-screening")
    ) and name.startswith("formal_screening_"):
        candidates.append("formal-screen")
    if any(
        token in reason_key for token in ("dimension coding", "dimension adjudication")
    ) and any(token in name for token in ("dimension", "construct")):
        candidates.append("derive-dimensions")
    if len(candidates) != 1:
        raise ProtocolError(
            f"cannot infer one stage from reason={reason!r}, artifact={path.name!r}"
        )
    return candidates[0]


def _identifier_columns(stage: str, fieldnames: Iterable[str]) -> list[str]:
    available = set(fieldnames)
    allowed = {
        "search-frame": ("query_id", "family_id"),
        "formal-screen": ("work_id",),
        "derive-dimensions": ("mention_id", "dimension_id", "work_id"),
    }[stage]
    columns = [column for column in allowed if column in available]
    if not columns:
        raise ProtocolError(f"{stage} artifact lacks a recognized object identifier")
    return columns


def _object_ids(path: Path, stage: str) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ProtocolError(f"review artifact has no CSV header: {path}")
        columns = _identifier_columns(stage, reader.fieldnames)
        values = {
            row[column].strip()
            for row in reader
            for column in columns
            if row.get(column, "").strip()
        }
    if not values:
        raise ProtocolError(
            f"review artifact has no non-empty object identifiers: {path}"
        )
    return sorted(values), columns


def _protected_snapshot(row: sqlite3.Row) -> tuple[Any, ...]:
    return tuple(row[column] for column in PROTECTED_COLUMNS)


def _normalized_payload(row: sqlite3.Row) -> str:
    path = _artifact_path(row["evidence"])
    stage = _stage(row["reason"], path)
    object_ids, columns = _object_ids(path, stage)
    artifact_hash = file_hash(path)
    return canonical_json(
        {
            "artifact_matches_session_output": artifact_hash == row["output_hash"],
            "artifact_path": str(path),
            "artifact_sha256": artifact_hash,
            "object_id_columns": columns,
            "object_ids": object_ids,
            "provenance_normalizer": "review_session_provenance_v1",
            "stage": stage,
        }
    )


def normalize_review_session_provenance(
    database: Path, *, dry_run: bool = False
) -> dict[str, Any]:
    """Normalize only legacy path-valued evidence and fail atomically on uncertainty."""
    connection = sqlite3.connect(database.resolve())
    connection.row_factory = sqlite3.Row
    try:
        rows = list(connection.execute("SELECT * FROM review_sessions ORDER BY 1"))
        changes: list[tuple[str, str]] = []
        protected = {row["review_session_id"]: _protected_snapshot(row) for row in rows}
        for row in rows:
            payload = _existing_payload(row["evidence"])
            if payload and payload.get("stage") and payload.get("object_ids"):
                continue
            changes.append((_normalized_payload(row), row["review_session_id"]))
        if not dry_run:
            connection.executemany(
                "UPDATE review_sessions SET evidence=? WHERE review_session_id=?",
                changes,
            )
            after = list(connection.execute("SELECT * FROM review_sessions ORDER BY 1"))
            if any(
                protected[row["review_session_id"]] != _protected_snapshot(row)
                for row in after
            ):
                raise ProtocolError("protected review-session fields changed")
            connection.commit()
        return {
            "database": str(database.resolve()),
            "dry_run": dry_run,
            "normalized_count": len(changes),
            "review_session_ids": [session_id for _, session_id in changes],
        }
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(
        canonical_json(
            normalize_review_session_provenance(args.database, dry_run=args.dry_run)
        )
    )


if __name__ == "__main__":
    main()
