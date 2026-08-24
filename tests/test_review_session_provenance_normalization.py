"""Tests for conservative legacy review-session provenance normalization."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from innovation_impact_feature_selection.evidence_derived.core import ProtocolError
from innovation_impact_feature_selection.evidence_derived.normalize_review_session_provenance import (
    PROTECTED_COLUMNS,
    normalize_review_session_provenance,
)


def _database(tmp_path: Path, *, reason: str, artifact: Path) -> Path:
    database = tmp_path / "evidence.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE review_sessions (review_session_id TEXT PRIMARY KEY, run_id TEXT, "
        "reviewer_role TEXT, input_hash TEXT, output_hash TEXT, model_label TEXT, "
        "evidence TEXT, reason TEXT, created_at TEXT)"
    )
    connection.execute(
        "INSERT INTO review_sessions VALUES(?,?,?,?,?,?,?,?,?)",
        (
            "RS_1",
            "RUN_1",
            "Primary AI",
            "a" * 64,
            "b" * 64,
            "Codex",
            str(artifact),
            reason,
            "2026-08-20T00:00:00+00:00",
        ),
    )
    connection.commit()
    connection.close()
    return database


def _csv(path: Path, field: str, values: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field, "decision"])
        writer.writeheader()
        writer.writerows({field: value, "decision": "include"} for value in values)


def _row(database: Path) -> sqlite3.Row:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    row = connection.execute("SELECT * FROM review_sessions").fetchone()
    connection.close()
    assert row is not None
    return row


@pytest.mark.parametrize(
    ("filename", "reason", "identifier", "expected_stage"),
    [
        (
            "search_frame_primary_coded.csv",
            "Blind primary terminology coding",
            "family_id",
            "search-frame",
        ),
        (
            "formal_screening_primary.csv",
            "Blind formal literature screening",
            "work_id",
            "formal-screen",
        ),
        (
            "construct_mentions_primary.csv",
            "Blind dimension coding",
            "mention_id",
            "derive-dimensions",
        ),
    ],
)
def test_normalizes_stage_and_object_ids_without_changing_protected_fields(
    tmp_path: Path,
    filename: str,
    reason: str,
    identifier: str,
    expected_stage: str,
) -> None:
    artifact = tmp_path / filename
    _csv(artifact, identifier, ["B", "A", "A"])
    database = _database(tmp_path, reason=reason, artifact=artifact)
    before = _row(database)

    result = normalize_review_session_provenance(database)
    after = _row(database)
    payload = json.loads(after["evidence"])

    assert result["normalized_count"] == 1
    assert payload["stage"] == expected_stage
    assert payload["object_ids"] == ["A", "B"]
    assert payload["object_id_columns"] == [identifier]
    assert payload["artifact_matches_session_output"] is False
    assert tuple(before[column] for column in PROTECTED_COLUMNS) == tuple(
        after[column] for column in PROTECTED_COLUMNS
    )
    assert normalize_review_session_provenance(database)["normalized_count"] == 0


def test_ambiguous_stage_fails_closed_and_rolls_back(tmp_path: Path) -> None:
    artifact = tmp_path / "unexpected.csv"
    _csv(artifact, "work_id", ["W1"])
    database = _database(
        tmp_path, reason="Blind formal literature screening", artifact=artifact
    )
    evidence_before = _row(database)["evidence"]

    with pytest.raises(ProtocolError, match="cannot infer one stage"):
        normalize_review_session_provenance(database)

    assert _row(database)["evidence"] == evidence_before
