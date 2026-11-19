from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import coding
from common import (
    DATABASE_PATH,
    json_hash,
    read_json,
    sha256_file,
    utc_now,
    write_json,
)
from database import initialize, invalidate_stages, log_event, set_stage
from providers import query_definition_hash


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "provider_physical_split_overrides_v3.json"
DEFAULT_REPORT = (
    ROOT / "outputs" / "provider_physical_split_application_v3.json"
)


def _logical_ids(config: Mapping[str, Any]) -> List[str]:
    """Return the deterministic parent logical-query set."""
    physical_ids = [
        str(row["physical_query_id"])
        for row in config["observed_failures"]
    ]
    logical_ids = sorted(
        {physical_id.split("__P", maxsplit=1)[0] for physical_id in physical_ids}
    )
    if any(not logical_id.startswith("L") for logical_id in logical_ids):
        raise ValueError("All overrides must target formal logical queries")
    return logical_ids


def _frame_rows(
    connection: sqlite3.Connection,
) -> Dict[str, List[Dict[str, Any]]]:
    """Export the database-defined frame in deterministic order."""
    queries = {
        "domains": "SELECT * FROM search_domains ORDER BY search_domain_id",
        "logical_queries": (
            "SELECT * FROM logical_queries WHERE logical_query_id LIKE 'L%' "
            "ORDER BY logical_query_id"
        ),
        "physical_queries": (
            "SELECT * FROM physical_queries "
            "WHERE logical_query_id LIKE 'L%' "
            "ORDER BY physical_query_id"
        ),
    }
    return {
        label: [dict(row) for row in connection.execute(query)]
        for label, query in queries.items()
    }


def _delete_old_requests(
    connection: sqlite3.Connection,
    physical_ids: Sequence[str],
) -> None:
    """Delete stale request checkpoints before replacing request packaging."""
    if not physical_ids:
        return
    placeholders = ",".join("?" for _ in physical_ids)
    for table in (
        "seed_recall_query_checks",
        "query_hits",
        "query_runs",
    ):
        connection.execute(
            f"DELETE FROM {table} WHERE physical_query_id "
            f"IN ({placeholders})",
            tuple(physical_ids),
        )
    connection.execute(
        f"DELETE FROM physical_queries WHERE physical_query_id "
        f"IN ({placeholders})",
        tuple(physical_ids),
    )


def _rebuild_logical_requests(
    connection: sqlite3.Connection,
    logical_id: str,
    maximum_length: int,
) -> Dict[str, Any]:
    """Repackage one logical query without changing its semantics."""
    logical = connection.execute(
        "SELECT * FROM logical_queries WHERE logical_query_id = ?",
        (logical_id,),
    ).fetchone()
    if logical is None or logical["status"] != "active":
        raise RuntimeError(f"Override target is not active: {logical_id}")
    if logical["press_status"] != "pass":
        raise RuntimeError(f"Override target lacks PRESS pass: {logical_id}")
    terms = json.loads(logical["domain_terms_json"])
    object_terms = json.loads(logical["object_terms_json"])
    context_terms = json.loads(logical["context_terms_json"])
    old_rows = connection.execute(
        """
        SELECT physical_query_id, expression, filter_expression, query_hash
        FROM physical_queries
        WHERE logical_query_id = ?
        ORDER BY physical_query_id
        """,
        (logical_id,),
    ).fetchall()
    if not old_rows:
        raise RuntimeError(f"Override target has no physical query: {logical_id}")
    filters = {str(row["filter_expression"]) for row in old_rows}
    if len(filters) != 1:
        raise RuntimeError(f"Inconsistent filters for {logical_id}")
    chunks = coding._split_term_block(
        terms,
        object_terms,
        context_terms,
        maximum_length=maximum_length,
    )
    flattened = [term for chunk in chunks for term in chunk]
    if flattened != sorted(set(terms), key=coding.normalize_term):
        raise RuntimeError(f"Term-union preservation failed for {logical_id}")
    old_ids = [str(row["physical_query_id"]) for row in old_rows]
    _delete_old_requests(connection, old_ids)
    filter_expression = next(iter(filters))
    new_rows: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        physical_id = f"{logical_id}__P{index:03d}"
        expression = " AND ".join(
            (
                coding.or_block(chunk),
                coding.or_block(object_terms),
                coding.or_block(context_terms),
            )
        )
        if len(expression) > maximum_length:
            raise RuntimeError(
                f"Physical expression still exceeds limit: {physical_id}"
            )
        query_hash = query_definition_hash(expression, filter_expression)
        connection.execute(
            """
            INSERT INTO physical_queries(
                physical_query_id, logical_query_id, provider,
                expression, filter_expression, status, query_hash
            ) VALUES (?, ?, 'OpenAlex', ?, ?, 'active', ?)
            """,
            (
                physical_id,
                logical_id,
                expression,
                filter_expression,
                query_hash,
            ),
        )
        new_rows.append(
            {
                "physical_query_id": physical_id,
                "expression_length": len(expression),
                "query_hash": query_hash,
                "term_count": len(chunk),
            }
        )
    return {
        "logical_query_id": logical_id,
        "logical_query_hash": str(logical["query_hash"]),
        "logical_term_union_hash": json_hash(
            {"terms": sorted(set(terms), key=coding.normalize_term)}
        ),
        "old_physical_query_ids": old_ids,
        "new_physical_queries": new_rows,
        "press_status": str(logical["press_status"]),
        "semantic_change": False,
    }


def _current_frame(
    connection: sqlite3.Connection,
) -> sqlite3.Row:
    row = connection.execute(
        """
        SELECT * FROM search_frame_versions
        WHERE status = 'current'
        ORDER BY frame_version DESC LIMIT 1
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("No current search frame")
    return row


def apply_overrides(
    connection: sqlite3.Connection,
    config_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    """Apply audited provider-only request splitting and version the frame."""
    coding._assert_not_frozen(connection)
    config = read_json(config_path)
    config_sha256 = sha256_file(config_path)
    source = _current_frame(connection)
    if int(source["frame_version"]) != int(config["source_frame_version"]):
        raise RuntimeError("Provider split source-frame version mismatch")
    if str(source["frame_hash"]) != str(config["source_frame_hash"]):
        raise RuntimeError("Provider split source-frame hash mismatch")
    logical_ids = _logical_ids(config)
    maximum_length = int(config["maximum_physical_expression_length"])
    connection.execute("BEGIN IMMEDIATE")
    try:
        splits = [
            _rebuild_logical_requests(
                connection,
                logical_id,
                maximum_length,
            )
            for logical_id in logical_ids
        ]
        target_version = int(source["frame_version"]) + 1
        connection.execute(
            """
            UPDATE logical_queries SET query_version = ?
            WHERE logical_query_id LIKE 'L%'
            """,
            (target_version,),
        )
        rows = _frame_rows(connection)
        previous_body = json.loads(source["frame_json"])
        frame_body = {
            "frame_version": target_version,
            "search_frame_discovery_stop": previous_body.get(
                "search_frame_discovery_stop",
                {},
            ),
            "input_terms": previous_body["input_terms"],
            "press_query_revisions": previous_body.get(
                "press_query_revisions",
                [],
            ),
            **rows,
            "provider_physical_split_override": {
                "amendment_id": config["amendment_id"],
                "config_sha256": config_sha256,
                "maximum_physical_expression_length": maximum_length,
                "semantic_change": False,
                "splits": splits,
            },
        }
        frame_hash = json_hash(frame_body)
        active_counts = {
            "K": int(
                connection.execute(
                    "SELECT COUNT(*) FROM search_domains "
                    "WHERE status = 'active'"
                ).fetchone()[0]
            ),
            "Q": int(
                connection.execute(
                    "SELECT COUNT(*) FROM logical_queries "
                    "WHERE status = 'active' "
                    "AND logical_query_id LIKE 'L%'"
                ).fetchone()[0]
            ),
            "P": int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM physical_queries p
                    JOIN logical_queries l USING(logical_query_id)
                    WHERE p.status = 'active' AND l.status = 'active'
                      AND l.logical_query_id LIKE 'L%'
                    """
                ).fetchone()[0]
            ),
        }
        connection.execute(
            """
            UPDATE search_frame_versions
            SET status = 'superseded_by_provider_physical_split'
            WHERE status = 'current'
            """
        )
        connection.execute(
            """
            INSERT INTO search_frame_versions(
                frame_version, input_term_hash, frame_hash, counts_json,
                frame_json, status, derived_at
            ) VALUES (?, ?, ?, ?, ?, 'current', ?)
            """,
            (
                target_version,
                source["input_term_hash"],
                frame_hash,
                json.dumps(active_counts, sort_keys=True),
                json.dumps(
                    frame_body,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                utc_now(),
            ),
        )
        set_stage(
            connection,
            "search_frame_derived",
            "complete",
            {
                **active_counts,
                "frame_hash": frame_hash,
                "frame_version": target_version,
                "provider_physical_split_config_sha256": config_sha256,
                "semantic_change": False,
            },
        )
        invalidate_stages(
            connection,
            (
                "search_frame_validated",
                "search_frame_frozen",
                "formal_retrieval_complete",
                "literature_screened",
                "indicators_extracted",
                "dimensions_derived",
                "features_selected",
                "audit_complete",
            ),
            "provider-only physical queries were equivalently re-split",
        )
        log_event(
            connection,
            "search_frame_versions",
            "frame_version",
            str(target_version),
            {
                "action": "provider_physical_split",
                "config_sha256": config_sha256,
                "semantic_change": False,
            },
        )
        result = {
            "amendment_id": config["amendment_id"],
            "applied_at": utc_now(),
            "config_path": str(config_path.resolve()),
            "config_sha256": config_sha256,
            "source_frame_hash": str(source["frame_hash"]),
            "source_frame_version": int(source["frame_version"]),
            "target_frame_hash": frame_hash,
            "target_frame_version": target_version,
            **active_counts,
            "splits": splits,
        }
        write_json(report_path, result)
        snapshot_suffix = config_sha256[:16]
        coding._register_snapshot(
            connection,
            f"provider_physical_split_config_{snapshot_suffix}",
            config_path,
            "provider_request_packaging_amendment",
        )
        coding._register_snapshot(
            connection,
            (
                "provider_physical_split_application_"
                f"v{target_version}_{snapshot_suffix}"
            ),
            report_path,
            "provider_request_packaging_audit",
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return result


def main() -> None:
    """Run the provider split amendment against the v3 database."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        result = apply_overrides(
            connection,
            args.config.resolve(),
            args.report.resolve(),
        )
    finally:
        connection.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
