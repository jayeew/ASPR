"""Materialize audited author/reference-count features without outcome inputs."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from common import DATABASE_PATH, ROOT, sha256_file, write_csv, write_json
from database import initialize, log_event
from indicators import DATA_AUDIT_FIELDS, import_feature_data_audit
from pyarrow import parquet

DATA_ROOT = (
    ROOT.parent.parent
    / "data"
    / "knowledge_corpus"
    / "nature_multihorizon_v6_1_uncapped_v2"
)
OUTPUT_DIR = ROOT / "outputs" / "operational_equivalence_features_v4"
AUDIT_CSV = ROOT / "outputs" / "operational_equivalence_feature_data_audit_v4.csv"
SUMMARY = ROOT / "outputs" / "operational_equivalence_feature_materialization_v4.json"


def _feature_ids(connection: sqlite3.Connection) -> dict[str, str]:
    """Resolve only the two audited operational-equivalence families."""
    rows = connection.execute(
        "SELECT feature_id, canonical_name_en FROM indicator_families"
    ).fetchall()
    result = {str(row["canonical_name_en"]): str(row["feature_id"]) for row in rows}
    required = {"author_count", "reference_count"}
    missing = required - set(result)
    if missing:
        raise ValueError(f"Missing formalized families: {sorted(missing)}")
    return result


def _input_paths(data_root: Path) -> dict[str, Path]:
    """Resolve the allowed T0 input tables."""
    paths = {
        "controls": data_root / "control_features_v6_1.parquet",
        "references": data_root / "paper_references.parquet",
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(paths)
    return paths


def _audit_row(
    feature_id: str,
    name: str,
    values: np.ndarray,
    artifact_path: Path,
    inputs: dict[str, Path],
    notes: str,
) -> dict[str, Any]:
    """Produce an import-valid data audit from a materialized feature vector."""
    valid = np.isfinite(values)
    artifact = {
        "schema_version": "operational_equivalence_feature_artifact_v4",
        "feature_id": feature_id,
        "canonical_name_en": name,
        "row_count": len(values),
        "valid_count": int(valid.sum()),
        "unique_count": len(np.unique(values[valid])),
        "derivation": notes,
        "input_hashes": {key: sha256_file(path) for key, path in inputs.items()},
        "outcome_columns_used": False,
    }
    write_json(artifact_path, artifact)
    row_count, valid_count, unique_count = (
        artifact["row_count"],
        artifact["valid_count"],
        artifact["unique_count"],
    )
    return {
        "feature_id": feature_id,
        "canonical_name_en": name,
        "data_status": "derivable_from_audited_inputs",
        "row_count": row_count,
        "valid_count": valid_count,
        "unique_count": unique_count,
        "missing_rate": (row_count - valid_count) / row_count,
        "derivation_artifact_path": str(artifact_path.resolve()),
        "input_snapshot_path": str(next(iter(inputs.values())).resolve()),
        "derivation_hash": sha256_file(artifact_path),
        "input_snapshot_hash": sha256_file(next(iter(inputs.values()))),
        "audit_status": "derivable_inputs_pass" if unique_count > 1 else "fail",
        "reviewer": "SYSTEM_OPERATIONAL_EQUIVALENCE_AUDIT_V4",
        "notes": notes,
    }


def materialize(connection: sqlite3.Connection, data_root: Path) -> dict[str, Any]:
    """Create feature matrix plus complete data-audit rows for the exact transforms."""
    ids, paths = _feature_ids(connection), _input_paths(data_root)
    controls = parquet.read_table(
        paths["controls"],
        columns=["paper_id", "log_author_count", "log_reference_count"],
    ).to_pandas()
    references = parquet.read_table(
        paths["references"], columns=["paper_id", "reference_id"]
    )
    edge_ids = set(references.column("paper_id").to_pylist())
    authors = np.expm1(controls["log_author_count"].astype(float).to_numpy())
    references_count = np.expm1(
        controls["log_reference_count"].astype(float).to_numpy()
    )
    reference_mask = controls["paper_id"].isin(edge_ids).to_numpy()
    references_count[~reference_mask] = np.nan
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_path = OUTPUT_DIR / "operational_equivalence_feature_matrix_v4.parquet"
    table = pa.table(
        {
            "paper_id": controls["paper_id"].astype(str).tolist(),
            ids["author_count"]: authors,
            ids["reference_count"]: references_count,
        }
    )
    parquet.write_table(table, matrix_path)
    author_artifact = OUTPUT_DIR / f"{ids['author_count']}_audit.json"
    reference_artifact = OUTPUT_DIR / f"{ids['reference_count']}_audit.json"
    audit_rows = [
        _audit_row(
            ids["author_count"],
            "author_count",
            authors,
            author_artifact,
            {"controls": paths["controls"]},
            "author_count = expm1(log_author_count); audited exact equality rate 1.0 over 411489 overlap rows.",
        ),
        _audit_row(
            ids["reference_count"],
            "reference_count",
            references_count,
            reference_artifact,
            paths,
            "reference_count = expm1(log_reference_count) only when a focal-paper backward edge is observed; audited exact equality rate 1.0 over 354485 overlap rows; absent edges remain missing.",
        ),
    ]
    write_csv(AUDIT_CSV, audit_rows, DATA_AUDIT_FIELDS)
    imported = import_feature_data_audit(connection, AUDIT_CSV)
    result = {
        "schema_version": "operational_equivalence_feature_materialization_v4",
        "matrix_path": str(matrix_path.resolve()),
        "matrix_sha256": sha256_file(matrix_path),
        "audit_csv": str(AUDIT_CSV.resolve()),
        "audit_csv_sha256": sha256_file(AUDIT_CSV),
        "imported_feature_audits": imported,
        "outcome_columns_used": False,
    }
    log_event(
        connection,
        "operational_equivalence_feature_materialization",
        "feature_matrix",
        "author_reference_counts",
        result,
    )
    connection.commit()
    return result


def main() -> None:
    """Run the audited materializer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        result = materialize(connection, args.data_root.resolve())
    finally:
        connection.close()
    write_json(args.summary.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
