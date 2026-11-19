#!/usr/bin/env python3
"""Build a fail-closed canonical training source from audited EF mappings."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    from .core import canonical_json, file_hash
except ImportError:  # Direct execution from this directory.
    from core import canonical_json, file_hash  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE = ROOT / "outputs" / "evidence_derived.sqlite3"
DEFAULT_MATRIX = (
    ROOT.parent
    / "evidence_derived_v3"
    / "experiments"
    / "oof_feature_set_comparison_v3"
    / "outputs"
    / "uncapped_v2"
    / "indicator_matrix_221.parquet"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "mapped_training_source.parquet"
DEFAULT_MANIFEST = ROOT / "outputs" / "mapped_training_source.manifest.json"


class MappedTrainingSourceError(RuntimeError):
    """Raised when an audited mapping cannot be reproduced safely."""


@dataclass(frozen=True)
class MappingSpec:
    indicator_id: str
    canonical_name: str
    role: str
    maximum_information_time: str
    mapping_type: str
    fields: tuple[str, ...]
    derivation: str
    source_snapshot_hash: str


def _load_fields(raw: str, indicator_id: str) -> tuple[str, ...]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise MappedTrainingSourceError(
            f"Invalid fields_json for {indicator_id}"
        ) from error
    if not isinstance(value, list) or not value:
        raise MappedTrainingSourceError(
            f"fields_json must be a non-empty list for {indicator_id}"
        )
    fields = tuple(str(item).strip() for item in value)
    if any(not field for field in fields) or len(set(fields)) != len(fields):
        raise MappedTrainingSourceError(
            f"fields_json contains blank or duplicate fields for {indicator_id}"
        )
    return fields


def _load_mappings(database: Path) -> list[MappingSpec]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute("""
            SELECT f.indicator_id, f.canonical_name, f.role,
                   f.maximum_information_time, m.mapping_type, m.fields_json,
                   m.derivation, m.source_snapshot_hash
            FROM indicator_data_mapping AS m
            JOIN indicator_families AS f USING(indicator_id)
            WHERE m.mapping_type IN ('direct', 'derivable')
              AND m.audit_status = 'pass'
            ORDER BY f.canonical_name COLLATE NOCASE, f.indicator_id
            """).fetchall()
        eligible = connection.execute("""
            SELECT COUNT(*) FROM indicator_data_mapping
            WHERE mapping_type IN ('direct', 'derivable') AND audit_status = 'pass'
            """).fetchone()[0]
    except sqlite3.Error as error:
        raise MappedTrainingSourceError(
            f"Cannot read mapping database: {error}"
        ) from error
    finally:
        connection.close()
    if eligible != len(rows):
        raise MappedTrainingSourceError(
            "Eligible indicator_data_mapping row has no indicator_families row"
        )
    if not rows:
        raise MappedTrainingSourceError(
            "No audited direct/derivable mappings are available"
        )
    specs = [
        MappingSpec(
            indicator_id=str(row["indicator_id"]),
            canonical_name=str(row["canonical_name"]).strip(),
            role=str(row["role"]).strip(),
            maximum_information_time=str(row["maximum_information_time"]).strip(),
            mapping_type=str(row["mapping_type"]),
            fields=_load_fields(str(row["fields_json"]), str(row["indicator_id"])),
            derivation=str(row["derivation"]).strip(),
            source_snapshot_hash=str(row["source_snapshot_hash"]).strip(),
        )
        for row in rows
    ]
    _validate_mapping_names_and_time(specs)
    return specs


def _validate_mapping_names_and_time(specs: list[MappingSpec]) -> None:
    seen: dict[str, str] = {}
    for spec in specs:
        if not spec.canonical_name:
            raise MappedTrainingSourceError(
                f"Blank canonical_name for {spec.indicator_id}"
            )
        normalized = spec.canonical_name.casefold()
        if normalized == "paper_id" or normalized in seen:
            first = seen.get(normalized, "paper_id")
            raise MappedTrainingSourceError(
                f"Duplicate canonical_name: {first} and {spec.indicator_id}"
            )
        seen[normalized] = spec.indicator_id
        if spec.maximum_information_time != "T0":
            raise MappedTrainingSourceError(
                f"Future field rejected for {spec.indicator_id}: "
                f"maximum_information_time={spec.maximum_information_time!r}"
            )
        if spec.role.casefold() == "outcome":
            raise MappedTrainingSourceError(
                f"Outcome field rejected for {spec.indicator_id}"
            )


def _derivation_rule(spec: MappingSpec) -> str:
    raw = spec.derivation.strip().casefold()
    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise MappedTrainingSourceError(
                f"Invalid derivation JSON for {spec.indicator_id}"
            ) from error
        raw = str(payload.get("rule", "")).strip().casefold()
    raw = re.sub(r"[\s-]+", "_", raw)
    aliases = {
        "copy": "identity",
        "direct": "identity",
        "row_sum": "sum",
        "row_mean": "mean",
        "subtract": "difference",
        "safe_ratio": "ratio",
        "first_non_null": "coalesce",
    }
    return aliases.get(raw, raw)


def _numeric_columns(table: pa.Table, spec: MappingSpec) -> list[pa.ChunkedArray]:
    columns = [table[field] for field in spec.fields]
    if any(
        not (pa.types.is_integer(column.type) or pa.types.is_floating(column.type))
        for column in columns
    ):
        raise MappedTrainingSourceError(
            f"Numeric derivation requires numeric fields for {spec.indicator_id}"
        )
    return [pc.cast(column, pa.float64()) for column in columns]


def _derive(
    table: pa.Table, spec: MappingSpec
) -> tuple[pa.Array | pa.ChunkedArray, str]:
    if spec.mapping_type == "direct":
        if len(spec.fields) != 1:
            raise MappedTrainingSourceError(
                f"Direct mapping must have exactly one field for {spec.indicator_id}"
            )
        return table[spec.fields[0]], "identity"
    rule = _derivation_rule(spec)
    if rule == "identity" and len(spec.fields) == 1:
        return table[spec.fields[0]], rule
    if rule == "coalesce" and len(spec.fields) >= 1:
        try:
            return (
                pc.coalesce(  # type: ignore[attr-defined]
                    *[table[field] for field in spec.fields]
                ),
                rule,
            )
        except pa.ArrowInvalid as error:
            raise MappedTrainingSourceError(
                f"Incompatible coalesce fields for {spec.indicator_id}"
            ) from error
    if rule not in {"sum", "mean", "difference", "ratio"}:
        raise MappedTrainingSourceError(
            f"Unsupported derivation {spec.derivation!r} for {spec.indicator_id}"
        )
    if rule in {"difference", "ratio"} and len(spec.fields) != 2:
        raise MappedTrainingSourceError(
            f"Derivation {rule} requires exactly two fields for {spec.indicator_id}"
        )
    columns = _numeric_columns(table, spec)
    if rule in {"sum", "mean"}:
        result = columns[0]
        for column in columns[1:]:
            result = pc.add(result, column)  # type: ignore[attr-defined]
        if rule == "mean":
            result = pc.divide(  # type: ignore[attr-defined]
                result, pa.scalar(float(len(columns)))
            )
        return result, rule
    if rule == "difference":
        return pc.subtract(columns[0], columns[1]), rule  # type: ignore[attr-defined]
    denominator_is_zero = pc.equal(  # type: ignore[attr-defined]
        columns[1], pa.scalar(0.0)
    )
    ratio = pc.divide(columns[0], columns[1])  # type: ignore[attr-defined]
    return (
        pc.if_else(  # type: ignore[attr-defined]
            denominator_is_zero, pa.scalar(None, pa.float64()), ratio
        ),
        rule,
    )


def _validate_source(table: pa.Table, specs: list[MappingSpec]) -> None:
    if "paper_id" not in table.column_names:
        raise MappedTrainingSourceError("Source matrix is missing paper_id")
    paper_ids = table["paper_id"]
    if paper_ids.null_count:
        raise MappedTrainingSourceError("paper_id contains null values")
    if pc.count_distinct(paper_ids).as_py() != table.num_rows:  # type: ignore[attr-defined]
        raise MappedTrainingSourceError("paper_id contains duplicate values")
    available = set(table.column_names)
    missing = sorted({field for spec in specs for field in spec.fields} - available)
    if missing:
        raise MappedTrainingSourceError(
            f"Mapped source fields are missing: {', '.join(missing)}"
        )


def _atomic_write_parquet(table: pa.Table, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            version="2.6",
            use_dictionary=True,
            row_group_size=65_536,
            write_statistics=True,
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def build_mapped_training_source(
    database: Path,
    matrix: Path,
    output: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Build canonical columns from audited mappings without selecting features."""
    paths = (database.resolve(), matrix.resolve())
    for path in paths:
        if not path.is_file():
            raise MappedTrainingSourceError(f"Required input does not exist: {path}")
    database_hash = file_hash(paths[0])
    matrix_hash = file_hash(paths[1])
    specs = _load_mappings(paths[0])
    required = ["paper_id", *sorted({field for spec in specs for field in spec.fields})]
    try:
        available = set(pq.ParquetFile(paths[1]).schema_arrow.names)
        missing = sorted(set(required) - available)
        if missing:
            raise MappedTrainingSourceError(
                f"Mapped source fields are missing: {', '.join(missing)}"
            )
        source = pq.read_table(paths[1], columns=required)
    except (pa.ArrowInvalid, OSError) as error:
        raise MappedTrainingSourceError(
            f"Cannot read source matrix: {error}"
        ) from error
    _validate_source(source, specs)
    arrays: list[pa.Array | pa.ChunkedArray] = [source["paper_id"]]
    names = ["paper_id"]
    lineage: list[dict[str, Any]] = []
    for spec in specs:
        array, rule = _derive(source, spec)
        arrays.append(array)
        names.append(spec.canonical_name)
        lineage.append(
            {
                "canonical_name": spec.canonical_name,
                "indicator_id": spec.indicator_id,
                "mapping_type": spec.mapping_type,
                "source_fields": list(spec.fields),
                "derivation_rule": rule,
                "role": spec.role,
                "maximum_information_time": spec.maximum_information_time,
                "source_snapshot_hash": spec.source_snapshot_hash,
                "output_dtype": str(array.type),
            }
        )
    result = pa.table(arrays, names=names)
    if file_hash(paths[0]) != database_hash or file_hash(paths[1]) != matrix_hash:
        raise MappedTrainingSourceError("An input changed while the build was running")
    _atomic_write_parquet(result, output.resolve())
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "builder": "mapped_training_source",
        "selection_performed": False,
        "hgb_results_read": False,
        "database": {"path": str(paths[0]), "sha256": database_hash},
        "source_matrix": {
            "path": str(paths[1]),
            "sha256": matrix_hash,
            "row_count": source.num_rows,
        },
        "output": {
            "path": str(output.resolve()),
            "sha256": file_hash(output.resolve()),
            "row_count": result.num_rows,
            "column_count": result.num_columns,
            "columns": result.column_names,
        },
        "field_lineage": lineage,
    }
    manifest_path = manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = build_mapped_training_source(
        args.database, args.matrix, args.output, args.manifest
    )
    print(canonical_json(manifest["output"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
