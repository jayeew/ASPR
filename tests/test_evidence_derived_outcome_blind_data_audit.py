"""Focused fail-closed tests for the outcome-blind data audit."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from innovation_impact_feature_selection.evidence_derived.audit_outcome_blind_data import (
    OutcomeBlindDataAuditError,
    audit_outcome_blind_data,
)


def _database(
    tmp_path: Path,
    *,
    fields: list[str] | None = None,
    missing_rate: float = 0.0,
    unique_count: int = 3,
    near_constant: int = 0,
) -> Path:
    path = tmp_path / "evidence.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
        CREATE TABLE indicator_families (
          indicator_id TEXT PRIMARY KEY,
          canonical_name TEXT NOT NULL,
          role TEXT NOT NULL,
          maximum_information_time TEXT NOT NULL
        );
        CREATE TABLE indicator_data_mapping (
          indicator_id TEXT PRIMARY KEY,
          mapping_type TEXT NOT NULL,
          fields_json TEXT NOT NULL,
          derivation TEXT NOT NULL,
          source_snapshot_hash TEXT NOT NULL,
          coverage REAL,
          missing_rate REAL,
          unique_count INTEGER,
          near_constant INTEGER NOT NULL,
          audit_status TEXT NOT NULL
        );
        """)
    connection.execute(
        "INSERT INTO indicator_families VALUES(?,?,?,?)",
        ("I1", "safe T0 feature", "predictive", "T0"),
    )
    connection.execute(
        "INSERT INTO indicator_data_mapping VALUES(?,?,?,?,?,?,?,?,?,?)",
        (
            "I1",
            "direct",
            json.dumps(fields or ["EF0001"]),
            "identity",
            "snapshot",
            1.0 - missing_rate,
            missing_rate,
            unique_count,
            near_constant,
            "pending",
        ),
    )
    connection.commit()
    connection.close()
    return path


def _matrix(
    tmp_path: Path, *, duplicate_ids: bool = False, future: bool = False
) -> Path:
    path = tmp_path / "matrix.parquet"
    payload: dict[str, list[object]] = {
        "paper_id": ["P1", "P1", "P3"] if duplicate_ids else ["P1", "P2", "P3"],
        "EF0001": [1.0, 2.0, 3.0],
        "EF0002": [3.0, 4.0, 5.0],
    }
    if future:
        payload["future_target"] = [10.0, 20.0, 30.0]
    pq.write_table(pa.table(payload), path)
    return path


def _inventory(tmp_path: Path, *, wrong_unique: bool = False) -> Path:
    path = tmp_path / "inventory.csv"
    fields = [
        "matrix_field",
        "matrix_dtype",
        "row_count",
        "missing_count",
        "missing_rate",
        "unique_count",
        "legacy_feature_id",
        "legacy_name",
        "legacy_formula",
        "legacy_use",
        "no_selection_decision",
    ]
    rows = []
    for field in ("EF0001", "EF0002", "future_target"):
        rows.append(
            {
                "matrix_field": field,
                "matrix_dtype": "float",
                "row_count": "3",
                "missing_count": "0",
                "missing_rate": "0.0",
                "unique_count": "2" if wrong_unique and field == "EF0001" else "3",
                "legacy_feature_id": field,
                "legacy_name": field,
                "legacy_formula": "identity",
                "legacy_use": "data_mapping_candidate_only",
                "no_selection_decision": "true",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _contract(tmp_path: Path, *, future_value: str = "0") -> Path:
    path = tmp_path / "field_contract.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["feature_id", "uses_future_information"]
        )
        writer.writeheader()
        for field in ("EF0001", "EF0002", "future_target"):
            writer.writerow(
                {
                    "feature_id": field,
                    "uses_future_information": (
                        future_value if field == "EF0001" else "0"
                    ),
                }
            )
    return path


def _run(
    tmp_path: Path,
    database: Path,
    matrix: Path,
    *,
    inventory: Path | None = None,
    contract: Path | None = None,
) -> dict[str, object]:
    return audit_outcome_blind_data(
        database,
        matrix,
        inventory or _inventory(tmp_path),
        contract or _contract(tmp_path),
        tmp_path / "outcome_blind_data_audit.json",
    )


def test_pass_writes_exact_metadata_only_after_checks(tmp_path: Path) -> None:
    database = _database(tmp_path)
    report = _run(tmp_path, database, _matrix(tmp_path))

    connection = sqlite3.connect(database)
    value = connection.execute(
        "SELECT value_json FROM metadata WHERE key='outcome_blind_audit'"
    ).fetchone()
    connection.close()
    metadata = json.loads(value[0])
    assert report["status"] == "pass"
    assert report["outcome_columns_used"] is False
    assert metadata["status"] == "pass"
    assert metadata["outcome_columns_used"] is False
    assert metadata["input_hash"] == report["input_hash"]
    assert metadata["hgb_oof_results_read"] is False


def test_duplicate_paper_id_fails_and_clears_stale_pass(tmp_path: Path) -> None:
    database = _database(tmp_path)
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO metadata VALUES(?,?)",
        ("outcome_blind_audit", '{"status":"pass"}'),
    )
    connection.commit()
    connection.close()

    with pytest.raises(OutcomeBlindDataAuditError, match="audit failed"):
        _run(tmp_path, database, _matrix(tmp_path, duplicate_ids=True))

    connection = sqlite3.connect(database)
    value = connection.execute(
        "SELECT value_json FROM metadata WHERE key='outcome_blind_audit'"
    ).fetchone()
    connection.close()
    assert value is None
    saved = json.loads((tmp_path / "outcome_blind_data_audit.json").read_text())
    assert saved["status"] == "fail"
    assert any("paper_id is not unique" in item for item in saved["failures"])


def test_direct_mapping_with_multiple_fields_fails_closed(tmp_path: Path) -> None:
    database = _database(tmp_path, fields=["EF0001", "EF0002"])
    with pytest.raises(OutcomeBlindDataAuditError, match="audit failed"):
        _run(tmp_path, database, _matrix(tmp_path))
    saved = json.loads((tmp_path / "outcome_blind_data_audit.json").read_text())
    assert saved["checks"]["direct_mapping_single_field"] is False


@pytest.mark.parametrize(
    ("matrix_future", "contract_value", "expected"),
    [
        (True, "0", "Known outcome columns exist"),
        (False, "1", "uses_future_information"),
    ],
)
def test_future_or_outcome_evidence_fails_closed(
    tmp_path: Path,
    matrix_future: bool,
    contract_value: str,
    expected: str,
) -> None:
    database = _database(tmp_path)
    with pytest.raises(OutcomeBlindDataAuditError, match="audit failed"):
        _run(
            tmp_path,
            database,
            _matrix(tmp_path, future=matrix_future),
            contract=_contract(tmp_path, future_value=contract_value),
        )
    saved = json.loads((tmp_path / "outcome_blind_data_audit.json").read_text())
    assert any(expected in item for item in saved["failures"])


def test_inventory_and_database_statistics_must_match(tmp_path: Path) -> None:
    database = _database(tmp_path, unique_count=2)
    with pytest.raises(OutcomeBlindDataAuditError, match="audit failed"):
        _run(
            tmp_path,
            database,
            _matrix(tmp_path),
            inventory=_inventory(tmp_path, wrong_unique=True),
        )
    saved = json.loads((tmp_path / "outcome_blind_data_audit.json").read_text())
    assert any(
        "unique_count differs from inventory" in item for item in saved["failures"]
    )
    assert any("DB unique_count mismatch" in item for item in saved["failures"])
