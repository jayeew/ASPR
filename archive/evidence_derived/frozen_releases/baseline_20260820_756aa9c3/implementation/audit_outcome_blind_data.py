#!/usr/bin/env python3
"""Fail-closed outcome-blind audit for imported indicator data mappings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sqlite3
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    from .core import canonical_json, file_hash, sha256_text
except ImportError:  # Direct execution from this directory.
    from core import canonical_json, file_hash, sha256_text  # type: ignore[no-redef]


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
DEFAULT_INVENTORY = (
    ROOT / "outputs" / "reviews" / "available_matrix_field_inventory.csv"
)
DEFAULT_FIELD_CONTRACT = (
    ROOT.parent
    / "evidence_derived_v3"
    / "outputs_recovered_20260819"
    / "complete_indicator_library_v3.csv"
)
DEFAULT_REPORT = ROOT / "outputs" / "outcome_blind_data_audit.json"
MAPPING_TYPES = {"direct", "derivable", "unavailable"}
OUTCOME_PATTERNS = (
    r"\bfuture(?:_|\b)",
    r"\btarget(?:_|\b)",
    r"\brealized(?:_|\b)",
    r"\bcitation (?:count|rate|impact|frequency|percentile)\b",
    r"\brelative citation ratio\b",
    r"\baltmetric\b",
    r"\bmendeley\b",
    r"\breadership\b",
    r"\bdownload count\b",
    r"\btweet count\b",
    r"\bsocial media attention\b",
    r"\bpolicy uptake\b",
    r"\bsocietal impact\b",
)


class OutcomeBlindDataAuditError(RuntimeError):
    """Raised when any outcome-blind data check fails."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_fields(raw: str, indicator_id: str) -> tuple[str, ...]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise OutcomeBlindDataAuditError(
            f"Invalid fields_json for {indicator_id}"
        ) from error
    if not isinstance(value, list):
        raise OutcomeBlindDataAuditError(
            f"fields_json must be a list for {indicator_id}"
        )
    fields = tuple(str(item).strip() for item in value)
    if any(not item for item in fields) or len(set(fields)) != len(fields):
        raise OutcomeBlindDataAuditError(
            f"fields_json has blank or duplicate values for {indicator_id}"
        )
    return fields


def _load_database_rows(database: Path) -> list[dict[str, Any]]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute("""
            SELECT m.indicator_id, m.mapping_type, m.fields_json, m.derivation,
                   m.source_snapshot_hash, m.coverage, m.missing_rate,
                   m.unique_count, m.near_constant, m.audit_status,
                   f.canonical_name, f.role, f.maximum_information_time
            FROM indicator_data_mapping AS m
            LEFT JOIN indicator_families AS f USING(indicator_id)
            ORDER BY m.indicator_id
            """).fetchall()
    except sqlite3.Error as error:
        raise OutcomeBlindDataAuditError(
            f"Cannot read indicator mappings: {error}"
        ) from error
    finally:
        connection.close()
    if not rows:
        raise OutcomeBlindDataAuditError("No imported indicator_data_mapping rows")
    return [dict(row) for row in rows]


def _known_outcome_name(value: str) -> bool:
    normalized = re.sub(r"[-_/]+", " ", value.casefold())
    if re.search(r"\b(?:prior|backward|reference|cited reference)\b", normalized):
        return False
    return any(re.search(pattern, normalized) for pattern in OUTCOME_PATTERNS)


def _missing_mask(column: pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    mask = pc.is_null(column)
    if pa.types.is_floating(column.type):
        mask = pc.or_(mask, pc.is_nan(column))  # type: ignore[attr-defined]
    return mask


def _field_stats(column: pa.ChunkedArray, threshold: float) -> dict[str, Any]:
    missing_mask = _missing_mask(column)
    missing_count = int(pc.sum(pc.cast(missing_mask, pa.int64())).as_py() or 0)
    valid = pc.filter(column, pc.invert(missing_mask))  # type: ignore[attr-defined]
    valid_count = len(valid)
    unique_count = int(pc.count_distinct(valid).as_py()) if valid_count else 0
    dominant_rate = 1.0
    if valid_count:
        counts = pc.value_counts(valid)
        dominant_count = int(pc.max(counts.field("counts")).as_py())
        dominant_rate = dominant_count / valid_count
    return {
        "row_count": len(column),
        "missing_count": missing_count,
        "missing_rate": missing_count / len(column) if len(column) else 1.0,
        "unique_count": unique_count,
        "dominant_nonmissing_rate": dominant_rate,
        "near_constant": bool(unique_count <= 1 or dominant_rate >= threshold),
    }


def _inventory_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        field = row.get("matrix_field", "").strip()
        if not field or field in result:
            raise OutcomeBlindDataAuditError(
                f"Inventory has blank or duplicate matrix_field: {field!r}"
            )
        result[field] = row
    return result


def _future_contract(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        feature = row.get("feature_id", "").strip()
        value = row.get("uses_future_information", "").strip()
        if feature:
            result[feature].add(value)
    return result


def _check_mapping_contracts(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    mapped: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in rows:
        indicator = str(row["indicator_id"])
        mapping_type = str(row["mapping_type"])
        if mapping_type not in MAPPING_TYPES:
            failures.append(f"{indicator}: invalid mapping_type={mapping_type!r}")
            continue
        if row["canonical_name"] is None:
            failures.append(f"{indicator}: missing indicator_families row")
            continue
        try:
            fields = _load_fields(str(row["fields_json"]), indicator)
        except OutcomeBlindDataAuditError as error:
            failures.append(str(error))
            continue
        if mapping_type == "direct" and len(fields) != 1:
            failures.append(f"{indicator}: direct mapping must have exactly one field")
        if mapping_type in {"direct", "derivable"} and not fields:
            failures.append(f"{indicator}: mapped row has no fields")
        if mapping_type == "unavailable" and fields:
            failures.append(f"{indicator}: unavailable mapping has source fields")
        if mapping_type in {"direct", "derivable"}:
            if str(row["maximum_information_time"]) != "T0":
                failures.append(f"{indicator}: maximum_information_time is not T0")
            if str(row["role"]).casefold() == "outcome":
                failures.append(f"{indicator}: outcome role cannot be mapped")
            names = [str(row["canonical_name"]), *fields]
            if any(_known_outcome_name(name) for name in names):
                failures.append(f"{indicator}: known outcome/future name detected")
            mapped.append({**row, "fields": fields})
    return mapped, failures


def _paper_id_check(matrix: Path) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    schema = pq.ParquetFile(matrix).schema_arrow
    if "paper_id" not in schema.names:
        return {"present": False}, ["Source matrix is missing paper_id"]
    ids = pq.read_table(matrix, columns=["paper_id"])["paper_id"]
    distinct = int(pc.count_distinct(ids).as_py())
    if ids.null_count:
        failures.append("paper_id contains null values")
    if distinct != len(ids):
        failures.append("paper_id is not unique")
    return {
        "present": True,
        "row_count": len(ids),
        "null_count": ids.null_count,
        "unique_count": distinct,
    }, failures


def _compare_number(actual: float, expected: Any, tolerance: float) -> bool:
    if expected is None:
        return False
    try:
        value = float(expected)
    except (TypeError, ValueError):
        return False
    return math.isclose(actual, value, rel_tol=0.0, abs_tol=tolerance)


def _audit_fields(
    matrix: Path,
    mapped: list[dict[str, Any]],
    inventory: dict[str, dict[str, str]],
    future_contract: dict[str, set[str]],
    near_constant_threshold: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    matrix_columns = set(pq.ParquetFile(matrix).schema_arrow.names)
    required = sorted({field for row in mapped for field in row["fields"]})
    missing = sorted(set(required) - matrix_columns)
    if missing:
        failures.append(f"Mapped fields absent from source matrix: {missing}")
    non_ef = sorted(
        name
        for name in matrix_columns
        if name != "paper_id" and not re.fullmatch(r"EF\d{4}", name)
    )
    leaking_columns = sorted(name for name in non_ef if _known_outcome_name(name))
    if leaking_columns:
        failures.append(
            f"Known outcome columns exist in source matrix: {leaking_columns}"
        )
    readable = [field for field in required if field in matrix_columns]
    table = pq.read_table(matrix, columns=readable)
    stats_rows: list[dict[str, Any]] = []
    db_by_field: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mapped:
        for field in row["fields"]:
            db_by_field[field].append(row)
    for field in readable:
        stats = _field_stats(table[field], near_constant_threshold)
        inv = inventory.get(field)
        if inv is None:
            failures.append(f"{field}: absent from available_matrix_field_inventory")
        else:
            if int(inv["row_count"]) != stats["row_count"]:
                failures.append(f"{field}: row_count differs from inventory")
            if int(inv["missing_count"]) != stats["missing_count"]:
                failures.append(f"{field}: missing_count differs from inventory")
            if not _compare_number(stats["missing_rate"], inv["missing_rate"], 5e-10):
                failures.append(f"{field}: missing_rate differs from inventory")
            if int(inv["unique_count"]) != stats["unique_count"]:
                failures.append(f"{field}: unique_count differs from inventory")
        contract_values = future_contract.get(field, set())
        if contract_values != {"0"}:
            failures.append(
                f"{field}: uses_future_information contract must be exactly {{'0'}}, got {sorted(contract_values)}"
            )
        for row in db_by_field[field]:
            indicator = str(row["indicator_id"])
            if not _compare_number(stats["missing_rate"], row["missing_rate"], 5e-10):
                failures.append(f"{indicator}/{field}: DB missing_rate mismatch")
            if (
                row["unique_count"] is None
                or int(row["unique_count"]) != stats["unique_count"]
            ):
                failures.append(f"{indicator}/{field}: DB unique_count mismatch")
            if int(row["near_constant"]) != int(stats["near_constant"]):
                failures.append(f"{indicator}/{field}: DB near_constant mismatch")
            if row["coverage"] is not None and not _compare_number(
                1.0 - stats["missing_rate"], row["coverage"], 5e-10
            ):
                failures.append(f"{indicator}/{field}: DB coverage mismatch")
        stats_rows.append(
            {
                "matrix_field": field,
                **stats,
                "inventory_present": inv is not None,
                "uses_future_information_values": sorted(contract_values),
                "mapped_indicator_ids": sorted(
                    str(row["indicator_id"]) for row in db_by_field[field]
                ),
            }
        )
    return stats_rows, failures


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _set_pass_metadata(database: Path, payload: dict[str, Any]) -> None:
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "INSERT INTO metadata(key,value_json) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
            ("outcome_blind_audit", canonical_json(payload)),
        )
        connection.commit()
    except sqlite3.Error as error:
        connection.rollback()
        raise OutcomeBlindDataAuditError(
            f"Cannot write outcome_blind_audit metadata: {error}"
        ) from error
    finally:
        connection.close()


def _clear_stale_metadata(database: Path) -> None:
    connection = sqlite3.connect(database)
    try:
        connection.execute("DELETE FROM metadata WHERE key='outcome_blind_audit'")
        connection.commit()
    except sqlite3.Error as error:
        connection.rollback()
        raise OutcomeBlindDataAuditError(
            f"Cannot clear stale outcome_blind_audit metadata: {error}"
        ) from error
    finally:
        connection.close()


def audit_outcome_blind_data(
    database: Path,
    matrix: Path,
    inventory_path: Path,
    field_contract_path: Path,
    report_path: Path,
    *,
    near_constant_threshold: float = 0.99,
) -> dict[str, Any]:
    """Audit imported mappings and write pass metadata only after every check passes."""
    paths = [
        database.resolve(),
        matrix.resolve(),
        inventory_path.resolve(),
        field_contract_path.resolve(),
    ]
    missing_inputs = [str(path) for path in paths if not path.is_file()]
    if missing_inputs:
        raise OutcomeBlindDataAuditError(f"Required inputs missing: {missing_inputs}")
    input_hashes = {path.name: file_hash(path) for path in paths}
    input_hash = sha256_text(canonical_json(input_hashes))
    failures: list[str] = []
    stats_rows: list[dict[str, Any]] = []
    paper_id: dict[str, Any] = {}
    mapped: list[dict[str, Any]] = []
    try:
        database_rows = _load_database_rows(paths[0])
        mapped, mapping_failures = _check_mapping_contracts(database_rows)
        failures.extend(mapping_failures)
        paper_id, paper_failures = _paper_id_check(paths[1])
        failures.extend(paper_failures)
        inventory = _inventory_index(_read_csv(paths[2]))
        future_contract = _future_contract(_read_csv(paths[3]))
        stats_rows, field_failures = _audit_fields(
            paths[1], mapped, inventory, future_contract, near_constant_threshold
        )
        failures.extend(field_failures)
    except (
        OSError,
        KeyError,
        ValueError,
        pa.ArrowInvalid,
        OutcomeBlindDataAuditError,
    ) as error:
        failures.append(str(error))
    status = "pass" if not failures else "fail"
    report: dict[str, Any] = {
        "schema_version": 1,
        "audit": "outcome_blind_data",
        "status": status,
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "input_hash": input_hash,
        "inputs": input_hashes,
        "near_constant_threshold": near_constant_threshold,
        "outcome_columns_used": any("outcome" in item.casefold() for item in failures),
        "hgb_oof_results_read": False,
        "selection_performed": False,
        "counts": {
            "database_mapping_rows": len(locals().get("database_rows", [])),
            "audited_direct_or_derivable_rows": len(mapped),
            "audited_source_fields": len(stats_rows),
        },
        "checks": {
            "paper_id_unique": not any("paper_id" in item for item in failures),
            "mapped_fields_exist": not any(
                "absent from source matrix" in item for item in failures
            ),
            "direct_mapping_single_field": not any(
                "direct mapping" in item for item in failures
            ),
            "no_future_or_known_outcome_columns": not any(
                "outcome" in item.casefold() or "future" in item.casefold()
                for item in failures
            ),
            "uses_future_information_zero": not any(
                "uses_future_information" in item for item in failures
            ),
            "matrix_inventory_statistics_match": not any(
                "inventory" in item for item in failures
            ),
            "database_statistics_match": not any("DB " in item for item in failures),
        },
        "paper_id": paper_id,
        "field_statistics": stats_rows,
        "failures": sorted(set(failures)),
    }
    _atomic_json(report_path, report)
    if failures:
        _clear_stale_metadata(paths[0])
        raise OutcomeBlindDataAuditError(
            f"Outcome-blind data audit failed with {len(set(failures))} issue(s); "
            f"see {report_path.resolve()}"
        )
    report_hash = file_hash(report_path.resolve())
    metadata = {
        "status": "pass",
        "outcome_columns_used": False,
        "input_hash": input_hash,
        "input_hashes": input_hashes,
        "audit_report_path": str(report_path.resolve()),
        "audit_report_sha256": report_hash,
        "near_constant_threshold": near_constant_threshold,
        "hgb_oof_results_read": False,
    }
    _set_pass_metadata(paths[0], metadata)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--field-contract", type=Path, default=DEFAULT_FIELD_CONTRACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--near-constant-threshold", type=float, default=0.99)
    args = parser.parse_args()
    try:
        report = audit_outcome_blind_data(
            args.database,
            args.matrix,
            args.inventory,
            args.field_contract,
            args.report,
            near_constant_threshold=args.near_constant_threshold,
        )
    except OutcomeBlindDataAuditError as error:
        parser.exit(1, f"{error}\n")
    print(
        canonical_json({"status": report["status"], "input_hash": report["input_hash"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
