from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List

from common import (
    DATABASE_PATH,
    sha256_file,
    utc_now,
    write_csv_iter,
    write_json,
)
from database import initialize, log_event, require_complete


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "human_tasks" / "fulltext_indicator_evidence_v3.csv"
)
DEFAULT_REPORT = ROOT / "outputs" / "fulltext_indicator_evidence_v3.json"
FIELDS = (
    "record_key",
    "doi",
    "title",
    "abstract",
    "work_type",
    "publication_year",
    "acquisition_status",
    "candidate_url",
    "local_path",
    "fulltext_sha256",
    "access_statement",
    "indicator_hint_count",
    "indicator_hints_json",
)


def _hints(
    connection: sqlite3.Connection,
    record_key: str,
) -> List[Dict[str, Any]]:
    """Return H2-included title/abstract hints without formula authority."""
    return [
        {
            "candidate_id": row["candidate_id"],
            "review_round": int(row["review_round"]),
            "raw_name_en": row["raw_name_en"],
            "location": row["location"],
            "evidence_span": row["evidence_span"],
            "proposed_role": row["proposed_role"],
            "canonical_family_label": row["canonical_family_label"],
        }
        for row in connection.execute(
            """
            SELECT candidate_id, review_round, raw_name_en, location,
                   evidence_span, proposed_role, canonical_family_label
            FROM discovery_indicator_candidates
            WHERE record_key = ? AND h2_decision = 'include'
            ORDER BY candidate_id
            """,
            (record_key,),
        )
    ]


def export(
    connection: sqlite3.Connection,
    output_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    """Export source-linked evidence for blind full-text indicator review."""
    require_complete(connection, ["literature_screened"])
    source_count = 0
    hint_count = 0
    with_hints = 0

    def rows() -> Iterable[Dict[str, Any]]:
        nonlocal source_count, hint_count, with_hints
        for row in connection.execute(
            """
            SELECT r.record_key, r.doi, r.title, r.abstract, r.work_type,
                   r.publication_year, a.status AS acquisition_status,
                   a.candidate_url, a.local_path,
                   a.sha256 AS fulltext_sha256, a.access_statement
            FROM records r
            JOIN screening_final s USING(record_key)
            LEFT JOIN fulltext_acquisitions a USING(record_key)
            WHERE s.final_decision = 'include'
              AND s.final_language = 'en'
            ORDER BY r.record_key
            """
        ):
            source_count += 1
            source_hints = _hints(connection, str(row["record_key"]))
            hint_count += len(source_hints)
            with_hints += int(bool(source_hints))
            yield {
                **dict(row),
                "indicator_hint_count": len(source_hints),
                "indicator_hints_json": json.dumps(
                    source_hints,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            }

    written = write_csv_iter(output_path, rows(), FIELDS)
    result = {
        "schema_version": "fulltext_indicator_evidence_v3",
        "sources": source_count,
        "sources_with_hints": with_hints,
        "h2_included_title_abstract_hints": hint_count,
        "formula_authority": False,
        "rows": written,
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "completed_at": utc_now(),
    }
    write_json(report_path, result)
    connection.execute(
        """
        INSERT INTO source_snapshots(
            source_id, path, sha256, role, imported_at
        ) VALUES (
            'fulltext_indicator_evidence_v3', ?, ?,
            'fulltext_indicator_review_evidence', ?
        )
        ON CONFLICT(source_id) DO UPDATE SET
            path = excluded.path,
            sha256 = excluded.sha256,
            role = excluded.role,
            imported_at = excluded.imported_at
        """,
        (
            str(output_path.resolve()),
            result["output_sha256"],
            utc_now(),
        ),
    )
    log_event(
        connection,
        "fulltext_indicator_evidence_export",
        "collection",
        "included_english_sources",
        result,
    )
    connection.commit()
    return result


def main() -> None:
    """Export the companion evidence table and its audit report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        result = export(
            connection,
            args.output.resolve(),
            args.report.resolve(),
        )
    finally:
        connection.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
