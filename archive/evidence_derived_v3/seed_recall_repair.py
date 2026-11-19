from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import coding
from common import (
    DATABASE_PATH,
    json_hash,
    normalize_text,
    normalize_term,
    read_csv,
    read_json,
    sha256_file,
    utc_now,
    write_csv,
    write_json,
)
from database import (
    initialize,
    invalidate_stages,
    log_event,
    set_stage,
    snapshot_import_file,
)
from providers import query_definition_hash


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "seed_recall_repair_candidates_v3.json"
H1_PROTOCOL = ROOT / "INDEPENDENT_CODEX_SEED_RECALL_REPAIR_H1_PROTOCOL_V3.json"
H2_PROTOCOL = ROOT / "INDEPENDENT_CODEX_SEED_RECALL_REPAIR_H2_PROTOCOL_V3.json"
H1_FIELDS = (
    "repair_id",
    "logical_query_id",
    "search_domain_id",
    "search_domain_label",
    "family_label",
    "current_domain_terms_json",
    "current_object_terms_json",
    "current_context_terms_json",
    "seed_ids_json",
    "seed_dois_json",
    "seed_sources_json",
    "reviewer_role",
    "decision",
    "domain_terms_json",
    "object_terms_json",
    "context_terms_json",
    "evidence_spans_json",
    "semantic_fit",
    "reason",
)
H2_FIELDS = (
    *H1_FIELDS[:11],
    "ai_domain_terms_json",
    "ai_object_terms_json",
    "ai_context_terms_json",
    "h1_decision",
    "h1_domain_terms_json",
    "h1_object_terms_json",
    "h1_context_terms_json",
    "h1_evidence_spans_json",
    "h1_semantic_fit",
    "h1_reason",
    "reviewer_role",
    "final_domain_terms_json",
    "final_object_terms_json",
    "final_context_terms_json",
    "final_evidence_spans_json",
    "concepts_complete",
    "boolean_logic_valid",
    "spelling_valid",
    "phrases_valid",
    "limits_justified",
    "construct_unchanged",
    "decision",
    "reason",
)


def _loads_list(value: Any, field: str) -> List[str]:
    """Parse a non-object JSON list of nonblank strings."""
    parsed = json.loads(str(value or "[]"))
    if not isinstance(parsed, list) or any(
        not isinstance(item, str) or not item.strip()
        for item in parsed
    ):
        raise ValueError(f"{field} must be a JSON string list")
    return [item.strip() for item in parsed]


def _bool(value: Any, field: str) -> bool:
    normalized = str(value or "").strip().casefold()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{field} must be true or false")
    return normalized == "true"


def _seed_sources(
    connection: sqlite3.Connection,
    seed_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    """Return source text that was available for one repair."""
    sources: List[Dict[str, Any]] = []
    for seed_id in seed_ids:
        seed = connection.execute(
            "SELECT * FROM evidence_seeds WHERE seed_id = ?",
            (seed_id,),
        ).fetchone()
        if seed is None:
            raise RuntimeError(f"Unknown repair seed: {seed_id}")
        record = connection.execute(
            """
            SELECT record_key, title, abstract, language, publication_year
            FROM records WHERE doi = ?
            ORDER BY length(abstract) DESC LIMIT 1
            """,
            (seed["doi"],),
        ).fetchone()
        if record is None:
            raise RuntimeError(f"Repair seed has no hydrated record: {seed_id}")
        sources.append(
            {
                "seed_id": seed_id,
                "doi": seed["doi"],
                "citation": seed["citation"],
                "record_key": record["record_key"],
                "title": record["title"],
                "abstract": record["abstract"],
                "language": record["language"],
                "publication_year": record["publication_year"],
            }
        )
    return sources


def _base_rows(
    connection: sqlite3.Connection,
    config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Resolve repair definitions against the current database frame."""
    rows: List[Dict[str, Any]] = []
    for repair in config["repairs"]:
        logical = connection.execute(
            """
            SELECT l.*, d.label AS search_domain_label
            FROM logical_queries l
            JOIN search_domains d USING(search_domain_id)
            WHERE logical_query_id = ?
            """,
            (repair["logical_query_id"],),
        ).fetchone()
        if logical is None or logical["status"] != "active":
            raise RuntimeError(
                f"Repair target is not active: {repair['logical_query_id']}"
            )
        sources = _seed_sources(connection, repair["seed_ids"])
        rows.append(
            {
                "repair_id": repair["repair_id"],
                "logical_query_id": repair["logical_query_id"],
                "search_domain_id": logical["search_domain_id"],
                "search_domain_label": logical["search_domain_label"],
                "family_label": logical["family_label"],
                "current_domain_terms_json": logical["domain_terms_json"],
                "current_object_terms_json": logical["object_terms_json"],
                "current_context_terms_json": logical["context_terms_json"],
                "seed_ids_json": json.dumps(repair["seed_ids"]),
                "seed_dois_json": json.dumps(
                    [source["doi"] for source in sources]
                ),
                "seed_sources_json": json.dumps(
                    sources,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
    return rows


def export_h1(
    connection: sqlite3.Connection,
    config_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Export a blind H1 worksheet without the AI term proposals."""
    config = read_json(config_path)
    rows = []
    for base in _base_rows(connection, config):
        rows.append(
            {
                **base,
                "reviewer_role": "H1",
                "decision": "",
                "domain_terms_json": "",
                "object_terms_json": "",
                "context_terms_json": "",
                "evidence_spans_json": "",
                "semantic_fit": "",
                "reason": "",
            }
        )
    write_csv(output_path, rows, H1_FIELDS)
    return {
        "rows": len(rows),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "protocol_sha256": sha256_file(H1_PROTOCOL),
    }


def _h1_by_repair(h1_path: Path) -> Dict[str, Dict[str, str]]:
    """Validate the minimally complete blind H1 artifact."""
    rows = {row["repair_id"]: row for row in read_csv(h1_path)}
    for repair_id, row in rows.items():
        if row["reviewer_role"].strip().upper() != "H1":
            raise ValueError(f"H1 role mismatch: {repair_id}")
        if row["decision"].strip().casefold() not in {
            "accept",
            "revise",
            "reject",
        }:
            raise ValueError(f"Invalid H1 decision: {repair_id}")
        if not row["reason"].strip():
            raise ValueError(f"H1 reason is required: {repair_id}")
        for field in (
            "domain_terms_json",
            "object_terms_json",
            "context_terms_json",
            "evidence_spans_json",
        ):
            _loads_list(row[field], f"{repair_id}/{field}")
        _bool(row["semantic_fit"], f"{repair_id}/semantic_fit")
        terms = [
            term
            for field in (
                "domain_terms_json",
                "object_terms_json",
                "context_terms_json",
            )
            for term in _loads_list(row[field], f"{repair_id}/{field}")
        ]
        _supported_terms(row, terms)
        _supported_spans(
            row,
            _loads_list(
                row["evidence_spans_json"],
                f"{repair_id}/evidence_spans_json",
            ),
        )
    return rows


def export_h2(
    connection: sqlite3.Connection,
    config_path: Path,
    h1_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Export AI/H1 comparisons with blank focused-PRESS adjudication."""
    config = read_json(config_path)
    h1_rows = _h1_by_repair(h1_path)
    repairs = {
        str(row["repair_id"]): row for row in config["repairs"]
    }
    rows = []
    for base in _base_rows(connection, config):
        repair_id = str(base["repair_id"])
        h1 = h1_rows.get(repair_id)
        if h1 is None:
            raise RuntimeError(f"Missing H1 repair row: {repair_id}")
        for field in H1_FIELDS[:11]:
            if str(h1[field]) != str(base[field]):
                raise RuntimeError(
                    f"H1 changed protected field {field}: {repair_id}"
                )
        ai = repairs[repair_id]
        row = {
            **base,
            "ai_domain_terms_json": json.dumps(ai["domain_terms"]),
            "ai_object_terms_json": json.dumps(ai["object_terms"]),
            "ai_context_terms_json": json.dumps(ai["context_terms"]),
            "h1_decision": h1["decision"],
            "h1_domain_terms_json": h1["domain_terms_json"],
            "h1_object_terms_json": h1["object_terms_json"],
            "h1_context_terms_json": h1["context_terms_json"],
            "h1_evidence_spans_json": h1["evidence_spans_json"],
            "h1_semantic_fit": h1["semantic_fit"],
            "h1_reason": h1["reason"],
            "reviewer_role": "H2",
        }
        for field in H2_FIELDS:
            row.setdefault(field, "")
        rows.append(row)
    write_csv(output_path, rows, H2_FIELDS)
    return {
        "rows": len(rows),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "protocol_sha256": sha256_file(H2_PROTOCOL),
    }


def _supported_terms(
    row: Mapping[str, str],
    terms: Iterable[str],
) -> None:
    """Require every final addition to occur in supplied source evidence."""
    sources = json.loads(row["seed_sources_json"])
    corpus = normalize_text(
        " ".join(
            str(source.get(field) or "")
            for source in sources
            for field in ("title", "abstract", "citation")
        )
    )
    for term in terms:
        if normalize_text(term) not in corpus:
            raise ValueError(
                f"Unsourced final repair term {term!r}: {row['repair_id']}"
            )


def _supported_spans(
    row: Mapping[str, str],
    spans: Sequence[str],
) -> None:
    """Require H2 evidence spans to be exact supplied-source substrings."""
    sources = json.loads(row["seed_sources_json"])
    source_values = [
        str(source.get(field) or "")
        for source in sources
        for field in ("title", "abstract", "citation")
    ]
    for span in spans:
        if not any(span in value for value in source_values):
            raise ValueError(
                f"Non-exact H2 evidence span {span!r}: {row['repair_id']}"
            )


def _merge_terms(existing: str, additions: Sequence[str]) -> List[str]:
    """Merge terms by normalized identity in deterministic order."""
    values = [*json.loads(existing), *additions]
    by_key: Dict[str, str] = {}
    for value in values:
        by_key.setdefault(normalize_term(value), str(value).strip())
    return [by_key[key] for key in sorted(by_key)]


def _delete_physical_rows(
    connection: sqlite3.Connection,
    physical_ids: Sequence[str],
) -> None:
    if not physical_ids:
        return
    placeholders = ",".join("?" for _ in physical_ids)
    for table in ("seed_recall_query_checks", "query_hits", "query_runs"):
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


def _apply_one(
    connection: sqlite3.Connection,
    row: Mapping[str, str],
) -> Dict[str, Any]:
    """Apply one H2-passed synonym repair to its existing logical family."""
    if row["reviewer_role"].strip().upper() != "H2":
        raise ValueError(f"H2 role mismatch: {row['repair_id']}")
    checks = [
        _bool(row[field], f"{row['repair_id']}/{field}")
        for field in (
            "concepts_complete",
            "boolean_logic_valid",
            "spelling_valid",
            "phrases_valid",
            "limits_justified",
            "construct_unchanged",
        )
    ]
    if row["decision"].strip().casefold() != "pass" or not all(checks):
        raise RuntimeError(f"Repair lacks focused PRESS pass: {row['repair_id']}")
    additions = {
        role: _loads_list(
            row[f"final_{role}_terms_json"],
            f"{row['repair_id']}/final_{role}_terms_json",
        )
        for role in ("domain", "object", "context")
    }
    all_terms = [
        term for values in additions.values() for term in values
    ]
    _supported_terms(row, all_terms)
    evidence_spans = _loads_list(
        row["final_evidence_spans_json"],
        f"{row['repair_id']}/final_evidence_spans_json",
    )
    if not evidence_spans or not row["reason"].strip():
        raise ValueError(
            f"H2 evidence spans and reason are required: {row['repair_id']}"
        )
    _supported_spans(row, evidence_spans)
    logical_id = row["logical_query_id"]
    logical = connection.execute(
        """
        SELECT l.*, d.label AS domain_label
        FROM logical_queries l
        JOIN search_domains d USING(search_domain_id)
        WHERE logical_query_id = ?
        """,
        (logical_id,),
    ).fetchone()
    old_physical = connection.execute(
        """
        SELECT physical_query_id, filter_expression
        FROM physical_queries WHERE logical_query_id = ?
        ORDER BY physical_query_id
        """,
        (logical_id,),
    ).fetchall()
    old_ids = [str(item["physical_query_id"]) for item in old_physical]
    old_recall_checks = [
        dict(item)
        for item in connection.execute(
            f"""
            SELECT * FROM seed_recall_query_checks
            WHERE physical_query_id IN (
                {','.join('?' for _ in old_ids)}
            )
            ORDER BY frame_version, physical_query_id, seed_set_hash
            """,
            tuple(old_ids),
        )
    ] if old_ids else []
    old_query_runs = [
        dict(item)
        for item in connection.execute(
            f"""
            SELECT * FROM query_runs
            WHERE physical_query_id IN (
                {','.join('?' for _ in old_ids)}
            )
            ORDER BY provider, physical_query_id, run_role
            """,
            tuple(old_ids),
        )
    ] if old_ids else []
    filters = {str(item["filter_expression"]) for item in old_physical}
    if logical is None or len(filters) != 1:
        raise RuntimeError(f"Invalid repair target state: {logical_id}")
    merged = {
        "domain": _merge_terms(
            logical["domain_terms_json"],
            additions["domain"],
        ),
        "object": _merge_terms(
            logical["object_terms_json"],
            additions["object"],
        ),
        "context": _merge_terms(
            logical["context_terms_json"],
            additions["context"],
        ),
    }
    expression = " AND ".join(
        (
            coding.or_block(merged["domain"]),
            coding.or_block(merged["object"]),
            coding.or_block(merged["context"]),
        )
    )
    logical_hash = json_hash(
        {
            "logical_query_id": logical_id,
            "domain": logical["domain_label"],
            "family": logical["family_label"],
            "expression": expression,
        }
    )
    connection.execute(
        """
        UPDATE logical_queries
        SET logical_expression = ?, domain_terms_json = ?,
            object_terms_json = ?, context_terms_json = ?,
            press_status = 'pass', press_reviewer = 'H2',
            press_notes = press_notes || ?, query_hash = ?
        WHERE logical_query_id = ?
        """,
        (
            expression,
            json.dumps(merged["domain"], ensure_ascii=False),
            json.dumps(merged["object"], ensure_ascii=False),
            json.dumps(merged["context"], ensure_ascii=False),
            f" | seed-recall repair {row['repair_id']} focused PRESS pass",
            logical_hash,
            logical_id,
        ),
    )
    connection.execute(
        """
        UPDATE press_reviews
        SET reviewer_role = 'H2', concepts_complete = 1,
            boolean_logic_valid = 1, spelling_valid = 1,
            phrases_valid = 1, limits_justified = 1,
            covered_by_logical_query_id = '',
            logical_coverage_verified = 0,
            result_set_coverage_verified = 0,
            independent_construct_role = 1,
            decision = 'pass', notes = ?, reviewed_at = ?
        WHERE logical_query_id = ?
        """,
        (row["reason"], utc_now(), logical_id),
    )
    _delete_physical_rows(connection, old_ids)
    filter_expression = next(iter(filters))
    chunks = coding._split_term_block(
        merged["domain"],
        merged["object"],
        merged["context"],
        maximum_length=1000,
    )
    new_ids = []
    for index, chunk in enumerate(chunks, start=1):
        physical_id = f"{logical_id}__P{index:03d}"
        physical_expression = " AND ".join(
            (
                coding.or_block(chunk),
                coding.or_block(merged["object"]),
                coding.or_block(merged["context"]),
            )
        )
        connection.execute(
            """
            INSERT INTO physical_queries(
                physical_query_id, logical_query_id, provider, expression,
                filter_expression, status, query_hash
            ) VALUES (?, ?, 'OpenAlex', ?, ?, 'active', ?)
            """,
            (
                physical_id,
                logical_id,
                physical_expression,
                filter_expression,
                query_definition_hash(
                    physical_expression,
                    filter_expression,
                ),
            ),
        )
        new_ids.append(physical_id)
    return {
        "repair_id": row["repair_id"],
        "logical_query_id": logical_id,
        "logical_query_hash": logical_hash,
        "old_physical_query_ids": old_ids,
        "old_seed_recall_checks": old_recall_checks,
        "old_query_runs": old_query_runs,
        "new_physical_query_ids": new_ids,
        "final_additions": additions,
    }


def _frame_rows(
    connection: sqlite3.Connection,
) -> Dict[str, List[Dict[str, Any]]]:
    queries = {
        "domains": "SELECT * FROM search_domains ORDER BY search_domain_id",
        "logical_queries": (
            "SELECT * FROM logical_queries WHERE logical_query_id LIKE 'L%' "
            "ORDER BY logical_query_id"
        ),
        "physical_queries": (
            "SELECT * FROM physical_queries "
            "WHERE logical_query_id LIKE 'L%' ORDER BY physical_query_id"
        ),
    }
    return {
        name: [dict(row) for row in connection.execute(query)]
        for name, query in queries.items()
    }


def apply_h2(
    connection: sqlite3.Connection,
    config_path: Path,
    h1_path: Path,
    h2_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    """Apply all independently reviewed repairs and create frame v+1."""
    coding._assert_not_frozen(connection)
    config = read_json(config_path)
    source = connection.execute(
        """
        SELECT * FROM search_frame_versions
        WHERE status = 'current'
        ORDER BY frame_version DESC LIMIT 1
        """
    ).fetchone()
    if (
        source is None
        or int(source["frame_version"]) != int(config["source_frame_version"])
        or str(source["frame_hash"]) != str(config["source_frame_hash"])
    ):
        raise RuntimeError("Seed-recall repair source frame mismatch")
    for path, role in ((h1_path, "H1"), (h2_path, "H2")):
        registered = connection.execute(
            """
            SELECT 1 FROM independent_ai_review_runs
            WHERE artifact_sha256 = ? AND reviewer_role = ?
              AND status = 'complete'
            """,
            (sha256_file(path), role),
        ).fetchone()
        if registered is None:
            raise RuntimeError(
                f"Unregistered independent {role} repair artifact: {path}"
            )
    h1_snapshot = snapshot_import_file(
        connection,
        h1_path,
        "seed_recall_repair_H1",
    )
    h2_snapshot = snapshot_import_file(
        connection,
        h2_path,
        "seed_recall_repair_H2",
    )
    connection.commit()
    h2_rows = read_csv(h2_snapshot)
    expected = {
        str(row["repair_id"]) for row in config["repairs"]
    }
    observed = {str(row["repair_id"]) for row in h2_rows}
    if observed != expected or len(h2_rows) != len(expected):
        raise RuntimeError("H2 repair artifact does not match repair set")
    connection.execute("BEGIN IMMEDIATE")
    try:
        applied = [_apply_one(connection, row) for row in h2_rows]
        target_version = int(source["frame_version"]) + 1
        connection.execute(
            """
            UPDATE logical_queries SET query_version = ?
            WHERE logical_query_id LIKE 'L%'
            """,
            (target_version,),
        )
        rows = _frame_rows(connection)
        previous = json.loads(source["frame_json"])
        frame_body = {
            "frame_version": target_version,
            "search_frame_discovery_stop": previous.get(
                "search_frame_discovery_stop",
                {},
            ),
            "input_terms": previous["input_terms"],
            "press_query_revisions": previous.get(
                "press_query_revisions",
                [],
            ),
            **rows,
            "seed_recall_repairs": {
                "config_sha256": sha256_file(config_path),
                "h1_sha256": sha256_file(h1_snapshot),
                "h2_sha256": sha256_file(h2_snapshot),
                "h1_protocol_sha256": sha256_file(H1_PROTOCOL),
                "h2_protocol_sha256": sha256_file(H2_PROTOCOL),
                "repairs": applied,
            },
        }
        frame_hash = json_hash(frame_body)
        counts = {
            "K": int(
                connection.execute(
                    """
                    SELECT COUNT(DISTINCT search_domain_id)
                    FROM logical_queries
                    WHERE status = 'active'
                      AND logical_query_id LIKE 'L%'
                    """
                ).fetchone()[0]
            ),
            "Q": int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM logical_queries
                    WHERE status = 'active'
                      AND logical_query_id LIKE 'L%'
                    """
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
            SET status = 'superseded_by_seed_recall_repair'
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
                json.dumps(counts, sort_keys=True),
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
                **counts,
                "frame_version": target_version,
                "frame_hash": frame_hash,
                "seed_recall_repairs": len(applied),
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
            "independently reviewed seed-recall term repairs changed queries",
        )
        result = {
            **counts,
            "source_frame_version": int(source["frame_version"]),
            "source_frame_hash": str(source["frame_hash"]),
            "target_frame_version": target_version,
            "target_frame_hash": frame_hash,
            "repairs": applied,
            "h1_sha256": sha256_file(h1_snapshot),
            "h2_sha256": sha256_file(h2_snapshot),
        }
        write_json(report_path, result)
        for source_id, path, role in (
            (
                "seed_recall_repair_candidates_v3",
                config_path,
                "seed_recall_repair_candidates",
            ),
            (
                "seed_recall_repair_h1_protocol_v3",
                H1_PROTOCOL,
                "independent_review_protocol",
            ),
            (
                "seed_recall_repair_h2_protocol_v3",
                H2_PROTOCOL,
                "independent_review_protocol",
            ),
            (
                "seed_recall_repair_application_v3",
                report_path,
                "seed_recall_repair_audit",
            ),
        ):
            coding._register_snapshot(connection, source_id, path, role)
        log_event(
            connection,
            "seed_recall_repair",
            "search_frame_version",
            str(target_version),
            result,
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return result


def main() -> None:
    """Export or apply the independently reviewed recall-repair workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    h1 = subparsers.add_parser("export-h1")
    h1.add_argument("--output", type=Path, required=True)
    h2 = subparsers.add_parser("export-h2")
    h2.add_argument("--h1", type=Path, required=True)
    h2.add_argument("--output", type=Path, required=True)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--h1", type=Path, required=True)
    apply_parser.add_argument("--h2", type=Path, required=True)
    apply_parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        if args.command == "export-h1":
            result = export_h1(
                connection,
                args.config.resolve(),
                args.output.resolve(),
            )
        elif args.command == "export-h2":
            result = export_h2(
                connection,
                args.config.resolve(),
                args.h1.resolve(),
                args.output.resolve(),
            )
        else:
            result = apply_h2(
                connection,
                args.config.resolve(),
                args.h1.resolve(),
                args.h2.resolve(),
                args.report.resolve(),
            )
    finally:
        connection.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
