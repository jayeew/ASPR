from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a local source file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_metadata(path: Path) -> Tuple[int | None, list[str] | None, str]:
    """Read Parquet row count and schema when PyArrow is available."""
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        return None, None, "skipped_pyarrow_unavailable"
    parquet_file = parquet.ParquetFile(path)
    return (
        int(parquet_file.metadata.num_rows),
        list(parquet_file.schema_arrow.names),
        "pass",
    )


def audit_sources(
    capabilities: Mapping[str, Any],
    verify_hashes: bool,
) -> Dict[str, Any]:
    """Resolve every declared source and optionally verify its digest."""
    results: Dict[str, Any] = {}
    failures = []
    source_schemas: Dict[str, set[str]] = {}
    for source_id, declaration in sorted(
        capabilities["source_files"].items()
    ):
        path = (ROOT / declaration["path"]).resolve()
        exists = path.is_file()
        actual_sha256 = sha256_file(path) if exists and verify_hashes else ""
        expected_sha256 = str(declaration["sha256"])
        matches = (
            actual_sha256 == expected_sha256 if verify_hashes and exists else None
        )
        row_count, columns, schema_status = (
            parquet_metadata(path)
            if exists
            else (None, None, "file_missing")
        )
        if columns is not None:
            source_schemas[source_id] = set(columns)
        result = {
            "path": str(path),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else 0,
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
            "sha256_matches": matches,
            "parquet_row_count": row_count,
            "parquet_column_count": len(columns) if columns is not None else None,
            "schema_status": schema_status,
        }
        results[source_id] = result
        if not exists or matches is False:
            failures.append(source_id)
    column_checks: Dict[str, Any] = {}
    missing_columns = []
    for column_name, declaration in sorted(
        capabilities["materialized_columns"].items()
    ):
        source_id = str(declaration["source"])
        source_columns = source_schemas.get(source_id)
        present = (
            column_name in source_columns
            if source_columns is not None
            else None
        )
        column_checks[column_name] = {
            "source": source_id,
            "present_in_frozen_parquet": present,
        }
        if present is False:
            missing_columns.append(column_name)
    expected_rows = int(capabilities["corpus"]["eligible_primary_articles"])
    primary_row_sources = (
        "control_features",
        "innovation_features",
        "opportunity_features",
        "primary_papers",
        "target_openalex_metadata",
    )
    row_count_mismatches = [
        source_id
        for source_id in primary_row_sources
        if results[source_id]["parquet_row_count"] not in (None, expected_rows)
    ]
    failures.extend(f"column:{name}" for name in missing_columns)
    failures.extend(f"row_count:{name}" for name in row_count_mismatches)
    return {
        "schema_version": "1.0.0",
        "verify_hashes": verify_hashes,
        "source_count": len(results),
        "all_sources_pass": not failures,
        "failed_source_ids": failures,
        "sources": results,
        "materialized_column_checks": column_checks,
        "missing_materialized_columns": missing_columns,
        "primary_row_count_mismatches": row_count_mismatches,
        "declared_materialized_column_count": len(
            capabilities["materialized_columns"]
        ),
        "declared_derivable_feature_count": len(
            capabilities["derivable_features"]
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Verify that the standalone feature census still points to the "
            "same frozen local source files."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "local_data_audit.json",
    )
    parser.add_argument(
        "--existence-only",
        action="store_true",
        help="Check paths but skip potentially expensive SHA-256 hashing.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the independent local-data provenance audit."""
    args = parse_args()
    capabilities = json.loads(
        (ROOT / "data_capabilities.json").read_text(encoding="utf-8")
    )
    result = audit_sources(
        capabilities,
        verify_hashes=not args.existence_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Audited {result['source_count']} frozen local sources; "
        f"all_sources_pass={result['all_sources_pass']}."
    )
    if not result["all_sources_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
