"""Build outcome-blind data-quality audit rows for v4 formula features."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict

import pyarrow.parquet as parquet

from common import DATABASE_PATH, sha256_file, write_csv, write_json
from database import initialize
from indicators import DATA_AUDIT_FIELDS


ROOT = Path(__file__).resolve().parent
MATRIX = ROOT / "outputs" / "evidence_features_v4.parquet"
REPORT = ROOT / "outputs" / "evidence_features_v4_report.json"
IMPLEMENTATION = ROOT / "materialize_evidence_features_v4.py"
DEFAULT_OUTPUT = ROOT / "outputs" / "data_audit_formula_matrix_v4.csv"
ARTIFACT_DIR = ROOT / "outputs" / "data_audit_formula_matrix_artifacts_v4"
COLUMNS = {
    "EF0002": "EF0002_complemented_gini_interdisciplinarity",
    "EF0004": "EF0004_gini_simpson_interdisciplinarity",
    "EF0007": "EF0007_rao_stirling_reference_diversity",
    "EF0008": "EF0008_reference_discipline_shannon_entropy",
    "EF0009": "EF0009_referenced_subject_category_variety",
}


def _unavailable(family: sqlite3.Row, note: str) -> Dict[str, object]:
    return {
        "feature_id": family["feature_id"],
        "canonical_name_en": family["canonical_name_en"],
        "data_status": "unavailable",
        "row_count": 0,
        "valid_count": 0,
        "unique_count": 0,
        "missing_rate": 0.0,
        "derivation_artifact_path": "",
        "input_snapshot_path": "",
        "derivation_hash": "",
        "input_snapshot_hash": "",
        "audit_status": "fail",
        "reviewer": "SYSTEM_V4_OUTCOME_BLIND_DATA_AUDIT",
        "notes": note,
    }


def build(connection: sqlite3.Connection, output: Path, artifact_dir: Path) -> None:
    """Write a complete nine-family audit without reading target outcomes."""
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    matrix_hash = sha256_file(MATRIX)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    schema = parquet.read_schema(MATRIX)
    rows = []
    for family in connection.execute(
        "SELECT feature_id, canonical_name_en, formula FROM indicator_families ORDER BY feature_id"
    ):
        feature_id = str(family["feature_id"])
        column = COLUMNS.get(feature_id)
        if column is None:
            rows.append(_unavailable(family, "No H2-eligible paper-level v4 formula matrix column."))
            continue
        if column not in schema.names:
            raise ValueError(f"Matrix lacks {column}")
        quality = report["feature_quality"][column]
        row_count = int(report["row_count"])
        valid_count = int(quality["valid_count"])
        unique_count = int(quality["unique_count"])
        artifact = {
            "schema_version": "evidence_derived_v4_formula_data_audit",
            "feature_id": feature_id,
            "canonical_name_en": family["canonical_name_en"],
            "formula": family["formula"],
            "column": column,
            "row_count": row_count,
            "valid_count": valid_count,
            "unique_count": unique_count,
            "missing_rate": (row_count - valid_count) / row_count,
            "outcome_columns_used": False,
            "implementation_path": str(IMPLEMENTATION.resolve()),
            "implementation_sha256": sha256_file(IMPLEMENTATION),
            "matrix_path": str(MATRIX.resolve()),
            "matrix_sha256": matrix_hash,
        }
        path = artifact_dir / f"{feature_id}_formula_matrix_audit.json"
        write_json(path, artifact)
        rows.append(
            {
                "feature_id": feature_id,
                "canonical_name_en": family["canonical_name_en"],
                "data_status": "materialized_audited",
                "row_count": row_count,
                "valid_count": valid_count,
                "unique_count": unique_count,
                "missing_rate": (row_count - valid_count) / row_count,
                "derivation_artifact_path": str(path.resolve()),
                "input_snapshot_path": str(MATRIX.resolve()),
                "derivation_hash": sha256_file(path),
                "input_snapshot_hash": matrix_hash,
                "audit_status": "pass" if unique_count > 1 else "fail",
                "reviewer": "SYSTEM_V4_OUTCOME_BLIND_DATA_AUDIT",
                "notes": "Raw-formula v4 matrix; the builder reads no outcome table or outcome column.",
            }
        )
    write_csv(output, rows, DATA_AUDIT_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        build(connection, args.output.resolve(), args.artifact_dir.resolve())
    finally:
        connection.close()
    print(json.dumps({"output": str(args.output.resolve()), "sha256": sha256_file(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
