"""Recover source leads for the v3 indicator-coverage benchmark.

The recovered v3 archive preserves candidate-family membership but not a
complete per-family source/formula trail.  This module creates bounded,
resumable English OpenAlex *coverage probes* for those labels.  A probe is a
discovery lead only: its records are never formal v4 search queries, formula
evidence, human decisions, or feature approvals.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from common import DATABASE_PATH, json_hash, sha256_file, utc_now, write_csv, write_json
from database import initialize, log_event
from providers import (
    fetch_json,
    insert_openalex_record,
    openalex_api_keys,
    openalex_record,
    openalex_url,
    safe_provider_error,
)

ROOT = Path(__file__).resolve().parent
ANCHOR = ROOT / "outputs" / "v3_coverage_anchor" / "complete_indicator_library_v3.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "v3_coverage_source_recovery_queue_v4.csv"
DEFAULT_SUMMARY = ROOT / "outputs" / "v3_coverage_source_recovery_summary_v4.json"
CUTOFF = "2026-07-28"
FILTER = f"language:en,from_publication_date:1900-01-01,to_publication_date:{CUTOFF}"
PER_PAGE = 25
FIELDS = (
    "v3_feature_id",
    "v3_canonical_name_en",
    "query_expression",
    "reported_total",
    "retrieved_rows",
    "probe_status",
    "candidate_record_keys_json",
    "next_action",
    "probe_note",
)


def read_anchor(path: Path) -> list[dict[str, str]]:
    """Read the immutable historical coverage benchmark."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 432:
        raise ValueError(f"Expected 432 v3 coverage labels, found {len(rows)}")
    required = {"feature_id", "canonical_name_en"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("v3 coverage anchor has no feature ID/name columns")
    return rows


def query_body(item: dict[str, str]) -> dict[str, Any]:
    """Return one deterministic, label-exact English discovery probe."""
    expression = " ".join(item["canonical_name_en"].split())
    return {
        "v3_feature_id": item["feature_id"],
        "expression": expression,
        "filter_expression": FILTER,
        "per_page": PER_PAGE,
        "cutoff": CUTOFF,
        "purpose": "bounded_source_recovery_probe_not_formal_search",
    }


def register_queries(
    connection: sqlite3.Connection, items: Iterable[dict[str, str]]
) -> list[dict[str, str]]:
    """Register all labels before requests so interrupted work is resumable."""
    registered: list[dict[str, str]] = []
    for item in items:
        body = query_body(item)
        query_hash = json_hash(body)
        feature_id = str(item["feature_id"])
        connection.execute(
            """
            INSERT INTO v3_coverage_recovery_queries(
                v3_feature_id, v3_canonical_name_en, query_expression,
                filter_expression, query_hash, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(v3_feature_id) DO UPDATE SET
                v3_canonical_name_en = excluded.v3_canonical_name_en,
                query_expression = excluded.query_expression,
                filter_expression = excluded.filter_expression,
                query_hash = excluded.query_hash,
                reported_total = CASE
                    WHEN v3_coverage_recovery_queries.query_hash
                         = excluded.query_hash
                    THEN v3_coverage_recovery_queries.reported_total
                    ELSE NULL
                END,
                retrieved_rows = CASE
                    WHEN v3_coverage_recovery_queries.query_hash
                         = excluded.query_hash
                    THEN v3_coverage_recovery_queries.retrieved_rows
                    ELSE 0
                END,
                complete = CASE
                    WHEN v3_coverage_recovery_queries.query_hash
                         = excluded.query_hash
                    THEN v3_coverage_recovery_queries.complete
                    ELSE 0
                END,
                error = CASE
                    WHEN v3_coverage_recovery_queries.query_hash
                         = excluded.query_hash
                    THEN v3_coverage_recovery_queries.error
                    ELSE ''
                END,
                updated_at = excluded.updated_at
            """,
            (
                feature_id,
                item["canonical_name_en"],
                body["expression"],
                body["filter_expression"],
                query_hash,
                utc_now(),
            ),
        )
        connection.execute(
            """
            INSERT INTO v3_coverage_reconciliation(
                v3_feature_id, v3_canonical_name_en
            ) VALUES (?, ?)
            ON CONFLICT(v3_feature_id) DO NOTHING
            """,
            (feature_id, item["canonical_name_en"]),
        )
        registered.append({**item, "query_hash": query_hash})
    connection.commit()
    return registered


def is_complete(connection: sqlite3.Connection, item: dict[str, str]) -> bool:
    """Return whether this exact bounded probe already has a terminal result."""
    row = connection.execute(
        """
        SELECT complete FROM v3_coverage_recovery_queries
        WHERE v3_feature_id = ? AND query_hash = ?
        """,
        (item["feature_id"], item["query_hash"]),
    ).fetchone()
    return row is not None and int(row["complete"]) == 1


def fetch_one(item: dict[str, str], api_key: str) -> dict[str, Any]:
    """Fetch one bounded OpenAlex probe without exposing its key."""
    url = openalex_url(
        expression=item["canonical_name_en"],
        filter_expression=FILTER,
        per_page=PER_PAGE,
        api_key=api_key,
    )
    try:
        payload = fetch_json(url)
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise ValueError("OpenAlex response has no results list")
        meta = payload.get("meta")
        total = meta.get("count") if isinstance(meta, dict) else None
        return {
            "item": item,
            "total": total if isinstance(total, int) else None,
            "results": raw_results,
            "error": "",
        }
    except Exception as error:  # provider helper has already bounded retries
        return {
            "item": item,
            "total": None,
            "results": [],
            "error": safe_provider_error(error),
        }


def persist_result(connection: sqlite3.Connection, result: dict[str, Any]) -> None:
    """Save probe metadata and candidate records transactionally."""
    item = result["item"]
    feature_id = str(item["feature_id"])
    error = str(result["error"])
    raw_results = result["results"]
    if not error:
        connection.execute(
            "DELETE FROM v3_coverage_recovery_hits WHERE v3_feature_id = ?",
            (feature_id,),
        )
        for rank, raw in enumerate(raw_results, start=1):
            if not isinstance(raw, dict):
                continue
            record = openalex_record(raw, f"v3_coverage_probe:{feature_id}")
            insert_openalex_record(connection, record)
            connection.execute(
                """
                INSERT OR REPLACE INTO v3_coverage_recovery_hits(
                    v3_feature_id, record_key, rank
                ) VALUES (?, ?, ?)
                """,
                (feature_id, record["record_key"], rank),
            )
    connection.execute(
        """
        UPDATE v3_coverage_recovery_queries
        SET reported_total = ?, retrieved_rows = ?, next_cursor = '',
            complete = ?, error = ?, updated_at = ?
        WHERE v3_feature_id = ? AND query_hash = ?
        """,
        (
            result["total"],
            len(raw_results),
            0 if error else 1,
            error,
            utc_now(),
            feature_id,
            item["query_hash"],
        ),
    )


def export_queue(connection: sqlite3.Connection, output: Path) -> dict[str, Any]:
    """Export the non-authorizing candidate-source queue and summary counts."""
    rows: list[dict[str, Any]] = []
    for row in connection.execute("""
        SELECT q.v3_feature_id, q.v3_canonical_name_en, q.query_expression,
               q.reported_total, q.retrieved_rows, q.complete, q.error,
               GROUP_CONCAT(h.record_key, char(31)) AS record_keys
        FROM v3_coverage_recovery_queries q
        LEFT JOIN v3_coverage_recovery_hits h USING(v3_feature_id)
        GROUP BY q.v3_feature_id
        ORDER BY q.v3_feature_id
        """):
        keys = [] if not row["record_keys"] else str(row["record_keys"]).split(chr(31))
        completed = int(row["complete"]) == 1
        status = "completed" if completed else "error_or_pending"
        rows.append(
            {
                "v3_feature_id": row["v3_feature_id"],
                "v3_canonical_name_en": row["v3_canonical_name_en"],
                "query_expression": row["query_expression"],
                "reported_total": row["reported_total"],
                "retrieved_rows": row["retrieved_rows"],
                "probe_status": status,
                "candidate_record_keys_json": json.dumps(keys, ensure_ascii=False),
                "next_action": (
                    "Independent source screen and fulltext/formula recovery; "
                    "a probe hit is not evidence of the named indicator."
                    if completed
                    else "Re-run this bounded probe; inspect safe error."
                ),
                "probe_note": (
                    "Bounded to the first 25 English OpenAlex results at the fixed cutoff; "
                    "not an official v4 logical or physical search query."
                ),
            }
        )
    write_csv(output, rows, FIELDS)
    return {
        "registered_labels": len(rows),
        "completed_probes": sum(row["probe_status"] == "completed" for row in rows),
        "failed_or_pending_probes": sum(
            row["probe_status"] != "completed" for row in rows
        ),
        "candidate_hits": sum(int(row["retrieved_rows"] or 0) for row in rows),
        "output_csv": str(output.resolve()),
        "output_csv_sha256": sha256_file(output),
    }


def run(
    connection: sqlite3.Connection,
    anchor: Path,
    output: Path,
    summary: Path,
    max_families: int | None,
) -> dict[str, Any]:
    """Register, retrieve, and export coverage probes with safe resumption."""
    registered = register_queries(connection, read_anchor(anchor))
    pending = [item for item in registered if not is_complete(connection, item)]
    if max_families is not None:
        if max_families < 1:
            raise ValueError("max_families must be positive")
        pending = pending[:max_families]
    keys = openalex_api_keys() or [""]
    with ThreadPoolExecutor(max_workers=min(len(keys), 2)) as executor:
        futures = [
            executor.submit(fetch_one, item, keys[index % len(keys)])
            for index, item in enumerate(pending)
        ]
        for future in as_completed(futures):
            persist_result(connection, future.result())
            connection.commit()
    report = export_queue(connection, output)
    report.update(
        {
            "schema_version": "v3_coverage_source_recovery_v4",
            "anchor_path": str(anchor.resolve()),
            "anchor_sha256": sha256_file(anchor),
            "fixed_cutoff": CUTOFF,
            "per_probe_result_cap": PER_PAGE,
            "newly_attempted_probes": len(pending),
            "coverage_probe_role": "source_lead_recovery_not_formal_k_q_p",
        }
    )
    write_json(summary, report)
    log_event(
        connection, "v3_coverage_source_recovery", "coverage_anchor", "v3_432", report
    )
    connection.commit()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--anchor", type=Path, default=ANCHOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--max-families", type=int)
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        result = run(
            connection,
            args.anchor.resolve(),
            args.output.resolve(),
            args.summary.resolve(),
            args.max_families,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
