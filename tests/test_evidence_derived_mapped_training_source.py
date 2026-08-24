"""Tests for the audited mapping-to-training-source boundary."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from innovation_impact_feature_selection.evidence_derived.build_mapped_training_source import (
    MappedTrainingSourceError,
    build_mapped_training_source,
)


def _database(tmp_path: Path) -> Path:
    database = tmp_path / "evidence.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript("""
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
          audit_status TEXT NOT NULL
        );
        """)
    connection.close()
    return database


def _add_mapping(
    database: Path,
    indicator_id: str,
    canonical_name: str,
    fields: list[str],
    *,
    mapping_type: str = "direct",
    derivation: str = "",
    role: str = "predictor",
    maximum_information_time: str = "T0",
    audit_status: str = "pass",
) -> None:
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO indicator_families VALUES(?,?,?,?)",
        (indicator_id, canonical_name, role, maximum_information_time),
    )
    connection.execute(
        "INSERT INTO indicator_data_mapping VALUES(?,?,?,?,?,?)",
        (
            indicator_id,
            mapping_type,
            json.dumps(fields),
            derivation,
            "snapshot-hash",
            audit_status,
        ),
    )
    connection.commit()
    connection.close()


def _matrix(tmp_path: Path, paper_ids: list[str] | None = None) -> Path:
    path = tmp_path / "matrix.parquet"
    pq.write_table(
        pa.table(
            {
                "paper_id": paper_ids or ["P1", "P2", "P3"],
                "EF1": [1, 2, 3],
                "EF2": [2.0, 4.0, 6.0],
                "EF3": [4.0, 8.0, 12.0],
                "future_target": [10, 20, 30],
            }
        ),
        path,
    )
    return path


def _build(tmp_path: Path, database: Path, matrix: Path) -> dict[str, object]:
    return build_mapped_training_source(
        database,
        matrix,
        tmp_path / "mapped.parquet",
        tmp_path / "mapped.manifest.json",
    )


def test_builds_stable_canonical_columns_and_lineage(tmp_path: Path) -> None:
    database = _database(tmp_path)
    _add_mapping(
        database,
        "I2",
        "zeta",
        ["EF2", "EF3"],
        mapping_type="derivable",
        derivation="row_mean",
    )
    _add_mapping(database, "I1", "Alpha", ["EF1"])
    _add_mapping(database, "I3", "ignored", ["future_target"], audit_status="fail")

    manifest = _build(tmp_path, database, _matrix(tmp_path))

    result = pq.read_table(tmp_path / "mapped.parquet")
    assert result.column_names == ["paper_id", "Alpha", "zeta"]
    assert result["Alpha"].to_pylist() == [1, 2, 3]
    assert result["zeta"].to_pylist() == [3.0, 6.0, 9.0]
    assert manifest["selection_performed"] is False
    assert manifest["hgb_results_read"] is False
    assert [row["source_fields"] for row in manifest["field_lineage"]] == [
        ["EF1"],
        ["EF2", "EF3"],
    ]
    saved = json.loads((tmp_path / "mapped.manifest.json").read_text())
    assert saved["output"]["sha256"] == manifest["output"]["sha256"]
    repeated = build_mapped_training_source(
        database,
        tmp_path / "matrix.parquet",
        tmp_path / "mapped-repeated.parquet",
        tmp_path / "mapped-repeated.manifest.json",
    )
    assert repeated["output"]["sha256"] == manifest["output"]["sha256"]


def test_rejects_missing_mapped_source_field(tmp_path: Path) -> None:
    database = _database(tmp_path)
    _add_mapping(database, "I1", "missing", ["EF404"])

    with pytest.raises(MappedTrainingSourceError, match="fields are missing.*EF404"):
        _build(tmp_path, database, _matrix(tmp_path))


def test_rejects_duplicate_canonical_name_case_insensitively(tmp_path: Path) -> None:
    database = _database(tmp_path)
    _add_mapping(database, "I1", "same", ["EF1"])
    _add_mapping(database, "I2", "SAME", ["EF2"])

    with pytest.raises(MappedTrainingSourceError, match="Duplicate canonical_name"):
        _build(tmp_path, database, _matrix(tmp_path))


def test_rejects_future_field(tmp_path: Path) -> None:
    database = _database(tmp_path)
    _add_mapping(
        database,
        "I1",
        "future",
        ["future_target"],
        maximum_information_time="T1",
    )

    with pytest.raises(MappedTrainingSourceError, match="Future field rejected"):
        _build(tmp_path, database, _matrix(tmp_path))


def test_rejects_outcome_field_even_at_t0(tmp_path: Path) -> None:
    database = _database(tmp_path)
    _add_mapping(database, "I1", "target", ["future_target"], role="outcome")

    with pytest.raises(MappedTrainingSourceError, match="Outcome field rejected"):
        _build(tmp_path, database, _matrix(tmp_path))


def test_rejects_multi_field_mapping_without_supported_rule(tmp_path: Path) -> None:
    database = _database(tmp_path)
    _add_mapping(
        database,
        "I1",
        "ambiguous",
        ["EF1", "EF2"],
        mapping_type="derivable",
        derivation="custom prose formula",
    )

    with pytest.raises(MappedTrainingSourceError, match="Unsupported derivation"):
        _build(tmp_path, database, _matrix(tmp_path))


def test_rejects_duplicate_paper_id(tmp_path: Path) -> None:
    database = _database(tmp_path)
    _add_mapping(database, "I1", "alpha", ["EF1"])

    with pytest.raises(MappedTrainingSourceError, match="paper_id contains duplicate"):
        _build(tmp_path, database, _matrix(tmp_path, ["P1", "P1", "P3"]))
