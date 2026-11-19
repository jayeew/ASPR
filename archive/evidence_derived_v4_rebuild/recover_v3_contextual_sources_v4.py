"""Run bounded contextual source recovery after H2 coverage-scope triage.

Unlike the initial label-only probe, this consumes H2's item-specific English
search terms and excludes H2 scope exclusions.  It still produces source
leads only, never formal-search K/Q/P queries or indicator approvals.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

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
OUTPUT = ROOT / "outputs" / "v3_contextual_source_recovery_queue_v4.csv"
SUMMARY = ROOT / "outputs" / "v3_contextual_source_recovery_summary_v4.json"
CUTOFF = "2026-07-28"
FILTER = f"language:en,from_publication_date:1900-01-01,to_publication_date:{CUTOFF}"
PER_PAGE = 25
FIELDS = (
    "v3_feature_id",
    "canonical_name_en",
    "h2_triage_decision",
    "h2_scope_role",
    "h2_search_terms_en",
    "query_expression",
    "reported_total",
    "retrieved_rows",
    "probe_status",
    "candidate_record_keys_json",
    "next_action",
)


def context_terms(scope_role: str) -> str:
    """Add a stable evidence context without changing the named construct."""
    if scope_role == "direct_innovation":
        return "scientific paper novelty bibliometric"
    return "scientific paper citation impact bibliometric"


def h2_candidates(connection: Any) -> list[dict[str, str]]:
    """Return only H2-routed source-recovery items from the audit ledger."""
    rows = connection.execute("""
        SELECT r.v3_feature_id, r.v3_canonical_name_en,
               t.triage_decision, t.scope_role_assessment, t.search_terms_en
        FROM v3_coverage_reconciliation r
        JOIN v3_coverage_triage_reviews t
          ON t.v3_feature_id = r.v3_feature_id AND t.reviewer_role = 'H2'
        WHERE t.triage_decision IN ('recover_priority', 'needs_source_evidence')
        ORDER BY r.v3_feature_id
        """).fetchall()
    if not rows:
        raise RuntimeError("No H2-adjudicated source-recovery candidates are imported")
    return [dict(row) for row in rows]


def expression(item: dict[str, str]) -> str:
    """Build a transparent H2-term plus fixed-context OpenAlex expression."""
    supplied = " ".join(str(item["search_terms_en"]).replace(";", " ").split())
    if not supplied:
        supplied = str(item["v3_canonical_name_en"])
    return f"{supplied} {context_terms(str(item['scope_role_assessment']))}"


def register(connection: Any, candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    """Register contextual probe definitions before requests for safe resumption."""
    registered: list[dict[str, str]] = []
    for item in candidates:
        query_expression = expression(item)
        query_hash = json_hash(
            {
                "strategy": "h2_contextual_source_recovery_v1",
                "feature_id": item["v3_feature_id"],
                "expression": query_expression,
                "filter": FILTER,
                "per_page": PER_PAGE,
            }
        )
        connection.execute(
            """
            INSERT INTO v3_contextual_recovery_queries(
                v3_feature_id, query_expression, filter_expression,
                query_hash, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(v3_feature_id) DO UPDATE SET
                query_expression = excluded.query_expression,
                filter_expression = excluded.filter_expression,
                reported_total = CASE WHEN v3_contextual_recovery_queries.query_hash
                    = excluded.query_hash THEN v3_contextual_recovery_queries.reported_total ELSE NULL END,
                retrieved_rows = CASE WHEN v3_contextual_recovery_queries.query_hash
                    = excluded.query_hash THEN v3_contextual_recovery_queries.retrieved_rows ELSE 0 END,
                complete = CASE WHEN v3_contextual_recovery_queries.query_hash
                    = excluded.query_hash THEN v3_contextual_recovery_queries.complete ELSE 0 END,
                error = CASE WHEN v3_contextual_recovery_queries.query_hash
                    = excluded.query_hash THEN v3_contextual_recovery_queries.error ELSE '' END,
                query_hash = excluded.query_hash, updated_at = excluded.updated_at
            """,
            (item["v3_feature_id"], query_expression, FILTER, query_hash, utc_now()),
        )
        registered.append(
            {**item, "query_expression": query_expression, "query_hash": query_hash}
        )
    connection.commit()
    return registered


def complete(connection: Any, item: dict[str, str]) -> bool:
    """Return whether this exact contextual probe has a saved terminal result."""
    row = connection.execute(
        "SELECT complete FROM v3_contextual_recovery_queries WHERE v3_feature_id = ? AND query_hash = ?",
        (item["v3_feature_id"], item["query_hash"]),
    ).fetchone()
    return row is not None and int(row["complete"]) == 1


def fetch(item: dict[str, str], api_key: str) -> dict[str, Any]:
    """Fetch a one-page contextual probe without emitting credentials."""
    try:
        payload = fetch_json(
            openalex_url(
                expression=item["query_expression"],
                filter_expression=FILTER,
                per_page=PER_PAGE,
                api_key=api_key,
            )
        )
        results = payload.get("results")
        meta = payload.get("meta")
        if not isinstance(results, list):
            raise ValueError("OpenAlex response has no results list")
        total = meta.get("count") if isinstance(meta, dict) else None
        return {
            "item": item,
            "results": results,
            "total": total if isinstance(total, int) else None,
            "error": "",
        }
    except Exception as error:
        return {
            "item": item,
            "results": [],
            "total": None,
            "error": safe_provider_error(error),
        }


def persist(connection: Any, result: dict[str, Any]) -> None:
    """Persist candidate records and their non-authorizing source-lead links."""
    item = result["item"]
    feature_id = str(item["v3_feature_id"])
    if not result["error"]:
        connection.execute(
            "DELETE FROM v3_contextual_recovery_hits WHERE v3_feature_id = ?",
            (feature_id,),
        )
        for rank, raw in enumerate(result["results"], start=1):
            if not isinstance(raw, dict):
                continue
            record = openalex_record(raw, f"v3_contextual_coverage_probe:{feature_id}")
            insert_openalex_record(connection, record)
            connection.execute(
                "INSERT OR REPLACE INTO v3_contextual_recovery_hits(v3_feature_id, record_key, rank) VALUES (?, ?, ?)",
                (feature_id, record["record_key"], rank),
            )
    connection.execute(
        """
        UPDATE v3_contextual_recovery_queries
        SET reported_total = ?, retrieved_rows = ?, complete = ?, error = ?, updated_at = ?
        WHERE v3_feature_id = ? AND query_hash = ?
        """,
        (
            result["total"],
            len(result["results"]),
            int(not result["error"]),
            result["error"],
            utc_now(),
            feature_id,
            item["query_hash"],
        ),
    )


def export(connection: Any, output: Path) -> dict[str, Any]:
    """Export all registered contextual source leads in deterministic ID order."""
    output_rows: list[dict[str, Any]] = []
    for query in connection.execute("""
        SELECT r.v3_feature_id, r.v3_canonical_name_en, t.triage_decision,
               t.scope_role_assessment, t.search_terms_en, q.query_expression,
               q.reported_total, q.retrieved_rows, q.complete, q.error
        FROM v3_contextual_recovery_queries q
        JOIN v3_coverage_reconciliation r USING(v3_feature_id)
        JOIN v3_coverage_triage_reviews t ON t.v3_feature_id = q.v3_feature_id
          AND t.reviewer_role = 'H2'
        ORDER BY r.v3_feature_id
        """):
        keys = [
            str(row[0])
            for row in connection.execute(
                "SELECT record_key FROM v3_contextual_recovery_hits WHERE v3_feature_id = ? ORDER BY rank, record_key",
                (query["v3_feature_id"],),
            )
        ]
        done = int(query["complete"]) == 1
        output_rows.append(
            {
                "v3_feature_id": query["v3_feature_id"],
                "canonical_name_en": query["v3_canonical_name_en"],
                "h2_triage_decision": query["triage_decision"],
                "h2_scope_role": query["scope_role_assessment"],
                "h2_search_terms_en": query["search_terms_en"],
                "query_expression": query["query_expression"],
                "reported_total": query["reported_total"],
                "retrieved_rows": query["retrieved_rows"],
                "probe_status": "completed" if done else "error_or_pending",
                "candidate_record_keys_json": json.dumps(keys, ensure_ascii=False),
                "next_action": (
                    "Independent title/abstract screen; source hits are not formula evidence or feature approvals."
                    if done
                    else "Resume contextual probe and inspect safe error."
                ),
            }
        )
    write_csv(output, output_rows, FIELDS)
    return {
        "candidate_labels": len(output_rows),
        "completed_probes": sum(
            row["probe_status"] == "completed" for row in output_rows
        ),
        "failed_or_pending_probes": sum(
            row["probe_status"] != "completed" for row in output_rows
        ),
        "candidate_hits": sum(int(row["retrieved_rows"] or 0) for row in output_rows),
        "output_csv": str(output.resolve()),
        "output_csv_sha256": sha256_file(output),
    }


def run(
    connection: Any, output: Path, summary: Path, maximum: int | None
) -> dict[str, Any]:
    """Run contextual probes with bounded parallelism and resumable persistence."""
    registered = register(connection, h2_candidates(connection))
    pending = [item for item in registered if not complete(connection, item)]
    if maximum is not None:
        if maximum < 1:
            raise ValueError("maximum must be positive")
        pending = pending[:maximum]
    keys = openalex_api_keys() or [""]
    with ThreadPoolExecutor(max_workers=min(2, len(keys))) as executor:
        futures = [
            executor.submit(fetch, item, keys[index % len(keys)])
            for index, item in enumerate(pending)
        ]
        for future in as_completed(futures):
            persist(connection, future.result())
            connection.commit()
    report = export(connection, output)
    report.update(
        {
            "schema_version": "v3_contextual_source_recovery_v4",
            "fixed_cutoff": CUTOFF,
            "per_probe_result_cap": PER_PAGE,
            "newly_attempted_probes": len(pending),
            "strategy": "h2_adjudicated_terms_plus_fixed_paper_level_context",
            "formal_k_q_p_changed": False,
            "selection_authorization": False,
        }
    )
    write_json(summary, report)
    log_event(
        connection, "v3_contextual_source_recovery", "coverage_anchor", "v3_432", report
    )
    connection.commit()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--maximum", type=int)
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        print(
            json.dumps(
                run(
                    connection,
                    args.output.resolve(),
                    args.summary.resolve(),
                    args.maximum,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
