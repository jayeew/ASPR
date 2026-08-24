from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Mapping

import pyarrow.parquet as parquet

from common import (
    DATABASE_PATH,
    normalize_term,
    read_json,
    sha256_file,
    write_csv,
    write_json,
)
from database import initialize
from indicators import DATA_AUDIT_FIELDS


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "feature_data_audit_completed_v3.csv"
DEFAULT_ARTIFACT_DIR = ROOT / "outputs" / "feature_data_audit_artifacts"


def _mapping_by_family(
    mapping: Mapping[str, Any],
    family: sqlite3.Row,
) -> Dict[str, Any] | None:
    """Resolve a mapping by feature ID first and canonical name second."""
    features = mapping.get("features")
    if not isinstance(features, dict):
        raise ValueError("Mapping JSON requires a features object")
    feature_id = str(family["feature_id"])
    exact = features.get(feature_id)
    if isinstance(exact, dict):
        return dict(exact)
    canonical_key = normalize_term(family["canonical_name_en"])
    for key, value in features.items():
        if (
            isinstance(value, dict)
            and normalize_term(str(key)) == canonical_key
        ):
            return dict(value)
    return None


def _source_path(
    mapping: Mapping[str, Any],
    feature_mapping: Mapping[str, Any],
) -> Path:
    """Resolve one named Parquet source from the mapping file."""
    sources = mapping.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("Mapping JSON requires a sources object")
    source_id = str(feature_mapping.get("source") or "").strip()
    raw_path = sources.get(source_id)
    if not source_id or not isinstance(raw_path, str):
        raise ValueError(f"Unknown mapped data source: {source_id}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _column_counts(path: Path, column_name: str) -> Dict[str, Any]:
    """Compute deterministic missingness and cardinality for one column."""
    schema = parquet.read_schema(path)
    if column_name not in schema.names:
        raise ValueError(f"Missing Parquet column {column_name}: {path}")
    series = parquet.read_table(path, columns=[column_name])[
        column_name
    ].to_pandas()
    valid = series.notna()
    row_count = int(len(series))
    valid_count = int(valid.sum())
    unique_count = int(series[valid].nunique(dropna=True))
    missing_rate = (
        (row_count - valid_count) / row_count if row_count else 0.0
    )
    return {
        "row_count": row_count,
        "valid_count": valid_count,
        "unique_count": unique_count,
        "missing_rate": missing_rate,
    }


def _unavailable_row(family: sqlite3.Row, reason: str) -> Dict[str, Any]:
    """Return an explicit failed data-audit disposition."""
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
        "reviewer": "SYSTEM_DATA_AUDIT_V3",
        "notes": reason,
    }


def build_audits(
    connection: sqlite3.Connection,
    mapping_path: Path,
    output_path: Path,
    artifact_dir: Path,
) -> Dict[str, Any]:
    """Build complete, import-ready local feature data audits."""
    mapping = read_json(mapping_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    hash_cache: Dict[Path, str] = {}
    rows = []
    mapped = 0
    for family in connection.execute(
        "SELECT * FROM indicator_families ORDER BY feature_id"
    ):
        feature_mapping = _mapping_by_family(mapping, family)
        if feature_mapping is None:
            rows.append(
                _unavailable_row(
                    family,
                    "No frozen local-data mapping for this canonical family.",
                )
            )
            continue
        source_path = _source_path(mapping, feature_mapping)
        column_name = str(feature_mapping.get("column") or "").strip()
        counts = _column_counts(source_path, column_name)
        if source_path not in hash_cache:
            hash_cache[source_path] = sha256_file(source_path)
        artifact = {
            "schema_version": "feature_data_audit_artifact_v3",
            "feature_id": family["feature_id"],
            "canonical_name_en": family["canonical_name_en"],
            "formula": family["formula"],
            "source_path": str(source_path),
            "source_sha256": hash_cache[source_path],
            "column": column_name,
            **counts,
            "outcome_blind": True,
            "maximum_information_time": family[
                "maximum_information_time"
            ],
        }
        artifact_path = (
            artifact_dir / f"{family['feature_id']}_audit.json"
        ).resolve()
        write_json(artifact_path, artifact)
        rows.append(
            {
                "feature_id": family["feature_id"],
                "canonical_name_en": family["canonical_name_en"],
                "data_status": "materialized_audited",
                **counts,
                "derivation_artifact_path": str(artifact_path),
                "input_snapshot_path": str(source_path),
                "derivation_hash": sha256_file(artifact_path),
                "input_snapshot_hash": hash_cache[source_path],
                "audit_status": (
                    "pass"
                    if counts["unique_count"] > 1
                    else "fail"
                ),
                "reviewer": "SYSTEM_DATA_AUDIT_V3",
                "notes": str(
                    feature_mapping.get("notes")
                    or (
                        "Exact frozen local Parquet column; outcome labels "
                        "were not used for availability or quality gating."
                    )
                ),
            }
        )
        if counts["unique_count"] <= 1:
            rows[-1] = _unavailable_row(
                family,
                "Mapped local column is constant or entirely missing.",
            )
        else:
            mapped += 1
    write_csv(output_path, rows, DATA_AUDIT_FIELDS)
    return {
        "rows": len(rows),
        "materialized_audited": mapped,
        "unavailable": len(rows) - mapped,
        "mapping_path": str(mapping_path),
        "mapping_sha256": sha256_file(mapping_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
    }


def main() -> None:
    """Build an import-ready audit CSV from a frozen feature mapping."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
    )
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        result = build_audits(
            connection,
            args.mapping.resolve(),
            args.output.resolve(),
            args.artifact_dir.resolve(),
        )
    finally:
        connection.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
