"""Audit whether included v4 works exist in the local OpenAlex snapshot index.

The snapshot index establishes local metadata coverage only.  It deliberately
does not claim that a PDF or formula-bearing full text is locally available.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from common import DATABASE_PATH, sha256_file, utc_now


DEFAULT_INDEX = Path(
    "/home/jayee/workspace/FabCitation/"
    "openalex_snapshot_reference_check_results/analysis_state.db"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / (
    "local_openalex_snapshot_coverage_v4.csv"
)
DEFAULT_REPORT = Path(__file__).resolve().parent / "outputs" / (
    "local_openalex_snapshot_coverage_v4.json"
)
FIELDS = ("record_key", "doi", "openalex_work_id", "snapshot_index_status")


def _work_id(raw_json: str) -> str:
    value = json.loads(raw_json).get("id")
    return str(value or "").rstrip("/").rsplit("/", maxsplit=1)[-1]


def audit(
    database: Path,
    index_database: Path,
    output: Path,
    report: Path,
) -> dict[str, Any]:
    """Write a deterministic coverage audit using indexed point lookups."""
    output.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(database)
    index = sqlite3.connect(f"file:{index_database}?mode=ro", uri=True)
    rows: list[dict[str, str]] = []
    try:
        records = source.execute(
            """
            SELECT r.record_key, r.doi, r.raw_json
            FROM records r JOIN screening_final s USING(record_key)
            WHERE s.final_decision = 'include' AND s.final_language = 'en'
            ORDER BY r.record_key
            """
        )
        for record_key, doi, raw_json in records:
            work_id = _work_id(str(raw_json))
            exists = index.execute(
                "SELECT 1 FROM works_index WHERE short_id = ?", (work_id,)
            ).fetchone()
            rows.append(
                {
                    "record_key": str(record_key),
                    "doi": str(doi or ""),
                    "openalex_work_id": work_id,
                    "snapshot_index_status": "found" if exists else "missing",
                }
            )
    finally:
        index.close()
        source.close()
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    found = sum(row["snapshot_index_status"] == "found" for row in rows)
    result = {
        "schema_version": "local_openalex_snapshot_coverage_v4",
        "included_english_records": len(rows),
        "snapshot_index_found": found,
        "snapshot_index_missing": len(rows) - found,
        "scope_note": (
            "The OpenAlex snapshot is metadata/link evidence, not a local "
            "PDF or formula-fulltext corpus."
        ),
        "output_path": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "completed_at": utc_now(),
    }
    report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    """Run the local snapshot coverage audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--index-database", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                args.database.resolve(),
                args.index_database.resolve(),
                args.output.resolve(),
                args.report.resolve(),
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
