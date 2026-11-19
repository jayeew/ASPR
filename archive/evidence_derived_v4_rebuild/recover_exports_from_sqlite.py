"""Validate a recovered evidence-v3 SQLite file and export the three CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Iterable


EXPORTS = {
    "complete_indicator_library_v3.csv": (
        "indicator_families",
        "feature_id",
        {"feature_id", "canonical_name_en"},
    ),
    "feature_gate_decisions_v3.csv": (
        "feature_decisions",
        "feature_id",
        {"feature_id", "gate_checks_json"},
    ),
    "candidate_dimensions_v3.csv": (
        "candidate_dimensions",
        "dimension_id",
        {"dimension_id", "label", "feature_ids_json"},
    ),
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a recovered file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    """Return a table's stored column order."""
    return [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]


def require(condition: bool, message: str) -> None:
    """Raise a descriptive error when one recovery invariant fails."""
    if not condition:
        raise ValueError(message)


def export_rows(
    connection: sqlite3.Connection,
    *,
    table: str,
    order_column: str,
    destination: Path,
) -> int:
    """Export one recovered SQLite table without transforming its values."""
    columns = table_columns(connection, table)
    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    query = f'SELECT {quoted_columns} FROM "{table}" ORDER BY "{order_column}"'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(connection.execute(query))
    return sum(1 for _ in connection.execute(f'SELECT 1 FROM "{table}"'))


def verify_membership(connection: sqlite3.Connection) -> None:
    """Check the cardinalities and frozen gate logic recorded in the run."""
    library_count = connection.execute("SELECT COUNT(*) FROM indicator_families").fetchone()[0]
    decision_count = connection.execute("SELECT COUNT(*) FROM feature_decisions").fetchone()[0]
    dimension_count = connection.execute("SELECT COUNT(*) FROM candidate_dimensions").fetchone()[0]
    require(library_count == 432, f"indicator_families count is {library_count}, not 432")
    require(decision_count == 432, f"feature_decisions count is {decision_count}, not 432")
    require(dimension_count == 66, f"candidate_dimensions count is {dimension_count}, not 66")
    rows = connection.execute("SELECT feature_id, gate_checks_json FROM feature_decisions").fetchall()
    gates = {feature_id: json.loads(value) for feature_id, value in rows}
    safe = (
        "G01_IN_SCOPE_ROLE",
        "G02_ARTICLE_LEVEL",
        "G05_PUBLICATION_TIME",
        "G06_NO_FUTURE_INFORMATION",
        "G08_BIAS_GUARDRAIL",
        "G09_NO_FATAL_VALIDITY_CONCERN",
        "G10_OUTCOME_BLIND_SELECTION",
    )
    broad = {feature_id for feature_id, checks in gates.items() if all(checks[key] for key in safe)}
    primary = {
        feature_id
        for feature_id, checks in gates.items()
        if feature_id in broad and checks["G03_PRIMARY_OR_FOUNDATIONAL_EVIDENCE"]
    }
    fulltext = {
        feature_id
        for feature_id, checks in gates.items()
        if feature_id in primary and checks["G13_ENGLISH_FULLTEXT_FORMULA_EVIDENCE"]
    }
    strict = {feature_id for feature_id, checks in gates.items() if all(checks.values())}
    require(tuple(map(len, (strict, fulltext, primary, broad))) == (7, 16, 154, 221), "frozen feature-set counts do not match")


def main() -> None:
    """Validate one candidate database and export the requested artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    connection = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        require(integrity == "ok", f"SQLite integrity check failed: {integrity}")
        for _, (table, _, required_columns) in EXPORTS.items():
            columns = set(table_columns(connection, table))
            require(required_columns.issubset(columns), f"{table} has wrong schema")
        verify_membership(connection)
        manifest = {"database_sha256": sha256(args.database), "exports": {}}
        for filename, (table, order_column, _) in EXPORTS.items():
            destination = args.output_dir / filename
            row_count = export_rows(
                connection,
                table=table,
                order_column=order_column,
                destination=destination,
            )
            manifest["exports"][filename] = {
                "rows": row_count,
                "sha256": sha256(destination),
            }
        (args.output_dir / "recovery_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
