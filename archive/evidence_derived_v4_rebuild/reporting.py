from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from common import (
    OUTPUT_DIR,
    ROOT,
    json_hash,
    normalize_doi,
    normalize_term,
    read_json,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_csv_iter,
    write_json,
)
from database import set_stage
from screening import screening_exclusion_counts

DISCOVERY_STOP_AMENDMENT_PATH = (
    ROOT / "protocol_amendment_round12_pragmatic_stop_v3.json"
)
ROUND12_EXTERNAL_REPORTING_CLARIFICATION_PATH = (
    ROOT / "protocol_amendment_round12_external_reporting_clarification_v3.json"
)
TERMINAL_FORMAL_COHORT_AMENDMENT_PATH = (
    ROOT / "protocol_amendment_round12_terminal_formal_cohort_v3.json"
)
FORMULA_OPERATIONALIZATION_AMENDMENT_PATH = (
    ROOT / "protocol_amendment_formula_operationalization_separation_v3.json"
)
TARGETED_FORMULA_COMPLETION_AMENDMENT_PATH = (
    ROOT / "protocol_amendment_targeted_formula_completion_v3.json"
)
FINAL_TRAINING_FEATURE_DIR = OUTPUT_DIR / "final_training_features_v3"
FINAL_TRAINING_MATRIX_PATH = (
    FINAL_TRAINING_FEATURE_DIR / "final_training_features_v3.parquet"
)
FINAL_TRAINING_SCHEMA_PATH = (
    FINAL_TRAINING_FEATURE_DIR / "final_training_features_schema_v3.json"
)


def _rows(
    connection: sqlite3.Connection,
    query: str,
    parameters: Sequence[Any] = (),
) -> list[dict[str, Any]]:
    return [
        dict(row) for row in connection.execute(query, tuple(parameters)).fetchall()
    ]


def _ensure_record_payload_digests(
    connection: sqlite3.Connection,
) -> dict[str, int]:
    """Backfill one compact digest per raw provider record exactly once."""
    missing = connection.execute("""
        SELECT r.provider, r.record_key, r.abstract,
               r.referenced_works_json, r.raw_json
        FROM records r
        LEFT JOIN record_payload_digests d
          ON d.provider = r.provider AND d.record_key = r.record_key
        WHERE d.record_key IS NULL
        """)
    inserted = 0
    for row in missing:
        payload = "\0".join(
            (
                str(row["abstract"]),
                str(row["referenced_works_json"]),
                str(row["raw_json"]),
            )
        )
        connection.execute(
            """
            INSERT INTO record_payload_digests(
                provider, record_key, payload_sha256
            ) VALUES (?, ?, ?)
            """,
            (
                row["provider"],
                row["record_key"],
                sha256_bytes(payload.encode("utf-8")),
            ),
        )
        inserted += 1
        if inserted % 2000 == 0:
            connection.commit()
    connection.commit()
    return {
        "inserted": inserted,
        "total": int(
            connection.execute(
                "SELECT COUNT(*) FROM record_payload_digests"
            ).fetchone()[0]
        ),
    }


def current_counts(connection: sqlite3.Connection) -> dict[str, int]:
    """Return the six non-quota counts at the current frozen state."""
    return {
        "K": int(connection.execute("""
                SELECT COUNT(DISTINCT search_domain_id)
                FROM logical_queries
                WHERE status = 'active'
                  AND logical_query_id LIKE 'L%'
                """).fetchone()[0]),
        "Q": int(connection.execute("""
                SELECT COUNT(*) FROM logical_queries
                WHERE status = 'active'
                  AND logical_query_id LIKE 'L%'
                """).fetchone()[0]),
        "P": int(connection.execute("""
                SELECT COUNT(*) FROM physical_queries
                WHERE status = 'active'
                  AND logical_query_id LIKE 'L%'
                """).fetchone()[0]),
        "M": int(
            connection.execute("SELECT COUNT(*) FROM candidate_dimensions").fetchone()[
                0
            ]
        ),
        "D": int(connection.execute("""
                SELECT COUNT(*) FROM dimension_decisions
                WHERE selected = 1 AND dimension_role = 'predictive'
                """).fetchone()[0]),
        "F": int(connection.execute("""
                SELECT COUNT(*) FROM feature_decisions
                WHERE final_role != 'excluded'
                """).fetchone()[0]),
    }


def _stage_summary(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in connection.execute("SELECT * FROM stage_status ORDER BY rowid"):
        result[str(row["stage"])] = {
            "status": row["status"],
            "details": json.loads(row["details_json"]),
        }
    return result


def _prisma(connection: sqlite3.Connection) -> dict[str, Any]:
    query_links = int(connection.execute("""
            SELECT COUNT(*) FROM query_hits
            WHERE provider = 'OpenAlex' AND run_role = 'formal'
            """).fetchone()[0])
    query_unique = int(connection.execute("""
            SELECT COUNT(DISTINCT record_key) FROM query_hits
            WHERE provider = 'OpenAlex' AND run_role = 'formal'
            """).fetchone()[0])
    terminal_cohort_links = int(connection.execute("""
            SELECT COUNT(*) FROM query_hits
            WHERE provider = 'OpenAlex' AND run_role = 'formal'
              AND rank BETWEEN 1 AND 10
            """).fetchone()[0])
    terminal_cohort_unique = int(connection.execute("""
            SELECT COUNT(DISTINCT record_key) FROM query_hits
            WHERE provider = 'OpenAlex' AND run_role = 'formal'
              AND rank BETWEEN 1 AND 10
            """).fetchone()[0])
    citation_unique = int(connection.execute("""
            SELECT COUNT(*) FROM records
            WHERE retrieval_route LIKE '%citation%'
            """).fetchone()[0])
    screened = int(
        connection.execute("SELECT COUNT(*) FROM screening_final").fetchone()[0]
    )
    included = int(connection.execute("""
            SELECT COUNT(*) FROM screening_final
            WHERE final_decision = 'include' AND final_language = 'en'
            """).fetchone()[0])
    return {
        "openalex_formal_query_record_links": query_links,
        "openalex_unique_formal_query_records": query_unique,
        "fixed_formal_cohort_query_record_links": (terminal_cohort_links),
        "fixed_formal_cohort_unique_records": terminal_cohort_unique,
        "total_unique_records_in_screening_scope": int(
            connection.execute("SELECT COUNT(*) FROM formal_review_records").fetchone()[
                0
            ]
        ),
        "duplicate_query_links_removed": query_links - query_unique,
        "unique_citation_route_records": citation_unique,
        "records_with_final_title_abstract_disposition": screened,
        "english_records_included_for_fulltext_indicator_census": included,
        "exclusions_by_reason": screening_exclusion_counts(connection),
    }


def _discovery_summary(connection: sqlite3.Connection) -> dict[str, Any]:
    """Summarize the pre-domain saturation sample without implying a census."""
    rows = connection.execute("""
        SELECT q.query_role, COUNT(*) AS strata,
               COALESCE(SUM(r.retrieved_rows), 0) AS retrieved_rows,
               COALESCE(SUM(r.pages), 0) AS pages,
               COALESCE(SUM(CASE WHEN r.complete = 1 THEN 1 ELSE 0 END), 0)
                   AS complete_strata
        FROM discovery_queries q
        LEFT JOIN discovery_query_runs r USING(discovery_query_id)
        WHERE q.status = 'active'
        GROUP BY q.query_role
        ORDER BY q.query_role
        """).fetchall()
    latest_round = connection.execute("""
        SELECT * FROM discovery_review_rounds
        ORDER BY iteration DESC LIMIT 1
        """).fetchone()
    unique_sample_records = int(connection.execute("""
            SELECT COUNT(DISTINCT h.record_key)
            FROM discovery_hits h
            JOIN discovery_queries q USING(discovery_query_id)
            WHERE q.status = 'active'
            """).fetchone()[0])
    unique_network_records = int(connection.execute("""
            SELECT COUNT(DISTINCT h.record_key)
            FROM discovery_hits h
            JOIN discovery_queries q USING(discovery_query_id)
            WHERE q.status = 'network'
            """).fetchone()[0])
    language_counts = {
        str(row["language"]): int(row["records"]) for row in connection.execute("""
            SELECT r.language, COUNT(DISTINCT r.record_key) AS records
            FROM records r
            JOIN discovery_hits h USING(record_key)
            GROUP BY r.language ORDER BY r.language
            """)
    }
    return {
        "retrieval_design": "deterministic_stratified_evidence_saturation",
        "broad_query_exhaustive": False,
        "active_sample_strata": sum(int(row["strata"]) for row in rows),
        "complete_sample_strata": sum(int(row["complete_strata"]) for row in rows),
        "within_stratum_rows_before_deduplication": sum(
            int(row["retrieved_rows"]) for row in rows
        ),
        "unique_deterministic_sample_records": unique_sample_records,
        "unique_development_citation_network_records": (unique_network_records),
        "unique_discovery_records": int(
            connection.execute(
                "SELECT COUNT(DISTINCT record_key) FROM discovery_hits"
            ).fetchone()[0]
        ),
        "assigned_review_records": int(connection.execute("""
                SELECT COUNT(DISTINCT record_key) FROM discovery_hits
                WHERE review_round > 0
                """).fetchone()[0]),
        "strata_by_role": [dict(row) for row in rows],
        "unique_records_by_openalex_language": language_counts,
        "latest_review_round": (
            dict(latest_round) if latest_round is not None else None
        ),
    }


def _deterministic_result_payload(
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    """Build a timestamp-free payload for reproducibility comparison."""
    tables = {
        "source_snapshots": (
            "SELECT source_id, sha256, role FROM source_snapshots " "ORDER BY source_id"
        ),
        "source_snapshot_supersessions": (
            "SELECT old_source_id, new_source_id, old_sha256, "
            "observed_current_sha256, authorization_source_id, reason "
            "FROM source_snapshot_supersessions ORDER BY old_source_id"
        ),
        "human_review_attestations": (
            "SELECT attestation_id, artifact_sha256, reviewer_role, "
            "reviewer_id, provenance_type, attestation_statement, "
            "attested_at, status, attestation_file_sha256 "
            "FROM human_review_attestations ORDER BY attestation_id"
        ),
        "independent_ai_review_runs": (
            "SELECT run_id, artifact_sha256, input_sha256, reviewer_role, "
            "reviewer_id, model, model_digest, prompt_sha256, "
            "parameters_json, item_count, status, manifest_sha256 "
            "FROM independent_ai_review_runs ORDER BY run_id"
        ),
        "review_run_supersessions": (
            "SELECT old_run_id, new_run_id, reason "
            "FROM review_run_supersessions ORDER BY old_run_id"
        ),
        "discovery_queries": (
            "SELECT * FROM discovery_queries ORDER BY discovery_query_id"
        ),
        "discovery_query_evidence": (
            "SELECT * FROM discovery_query_evidence " "ORDER BY discovery_query_id"
        ),
        "discovery_query_runs": (
            "SELECT discovery_query_id, query_hash, reported_sample_total, "
            "retrieved_rows, unique_hits, pages, next_page, complete, "
            "stopped_reason, error FROM discovery_query_runs "
            "ORDER BY discovery_query_id"
        ),
        "discovery_hits": (
            "SELECT discovery_query_id, record_key, sample_rank, "
            "selection_hash, review_rank, review_round, review_status "
            "FROM discovery_hits ORDER BY discovery_query_id, record_key"
        ),
        "discovery_review_rounds": (
            "SELECT iteration, saturation_phase, "
            "batch_first_rank, batch_last_rank, "
            "assigned_records, fully_reviewed, "
            "new_nonredundant_english_terms, "
            "new_canonical_indicator_families, consecutive_zero_rounds, "
            "reviewer_role, decision, stop_basis, "
            "protocol_amendment_id, protocol_amendment_sha256, notes "
            "FROM discovery_review_rounds ORDER BY iteration"
        ),
        "discovery_indicator_candidates": (
            "SELECT * FROM discovery_indicator_candidates " "ORDER BY candidate_id"
        ),
        "discovery_extraction_reviews": (
            "SELECT record_key, review_round, reviewer_role, "
            "extraction_complete, no_relevant_items, notes "
            "FROM discovery_extraction_reviews "
            "ORDER BY review_round, record_key, reviewer_role"
        ),
        "seeds": (
            "SELECT seed_id, doi, language, seed_role, supplied_by, "
            "hidden_during_development, eligibility_status, "
            "indexability_status, recall_status, recall_query_ids, "
            "nonrecall_reason FROM evidence_seeds "
            "ORDER BY seed_role, seed_id"
        ),
        "hidden_seed_search_log": (
            "SELECT search_run_id, reviewer_role, route, source_name, "
            "exact_query_or_seed, executed_at, retrieved_count, "
            "screened_count, eligible_seed_count, "
            "eligible_seed_dois_json, completion_status, notes "
            "FROM hidden_seed_search_log "
            "ORDER BY search_run_id"
        ),
        "terms": (
            "SELECT term_id, source_record_key, source_id, source_type, "
            "source_language_status, source_language_evidence, "
            "verbatim_term, normalized_term, match_key, location, "
            "evidence_span, proposed_role, status, exclusion_reason "
            "FROM raw_terms ORDER BY term_id"
        ),
        "term_coding": (
            "SELECT term_id, coder_role, canonical_term, "
            "term_family_label, term_relation, search_domain_label, "
            "search_domain_definition, query_family_label, cross_domain, "
            "decision, reason FROM term_coding ORDER BY term_id, coder_role"
        ),
        "term_families": ("SELECT * FROM term_families ORDER BY term_family_id"),
        "canonical_terms": ("SELECT * FROM canonical_terms ORDER BY canonical_term_id"),
        "search_frame_versions": (
            "SELECT frame_version, input_term_hash, frame_hash, "
            "counts_json, frame_json, status "
            "FROM search_frame_versions ORDER BY frame_version"
        ),
        "domains": ("SELECT * FROM search_domains ORDER BY search_domain_id"),
        "logical_queries": ("SELECT * FROM logical_queries ORDER BY logical_query_id"),
        "physical_queries": (
            "SELECT * FROM physical_queries "
            "WHERE logical_query_id LIKE 'L%' ORDER BY physical_query_id"
        ),
        "query_runs": (
            "SELECT provider, physical_query_id, run_role, query_hash, "
            "reported_total, retrieved_rows, unique_hits, pages, "
            "next_cursor, complete, stopped_reason, error "
            "FROM query_runs ORDER BY provider, physical_query_id, run_role"
        ),
        "query_hits": (
            "SELECT provider, physical_query_id, run_role, record_key, rank "
            "FROM query_hits ORDER BY provider, physical_query_id, "
            "run_role, record_key"
        ),
        "records": (
            "SELECT r.provider, r.record_key, r.provider_id, r.doi, r.title, "
            "r.language, r.publication_year, r.work_type, r.source_url, "
            "r.retrieval_route, d.payload_sha256 "
            "FROM records r JOIN record_payload_digests d "
            "ON d.provider = r.provider AND d.record_key = r.record_key "
            "ORDER BY r.record_key"
        ),
        "press_reviews": (
            "SELECT logical_query_id, reviewer_role, concepts_complete, "
            "boolean_logic_valid, spelling_valid, phrases_valid, "
            "limits_justified, covered_by_logical_query_id, "
            "logical_coverage_verified, result_set_coverage_verified, "
            "independent_construct_role, decision, notes "
            "FROM press_reviews ORDER BY logical_query_id"
        ),
        "screening_decisions": (
            "SELECT record_key, reviewer_role, language_judgment, "
            "language_evidence, decision, exclusion_reason, evidence_span, "
            "notes FROM screening_decisions "
            "ORDER BY record_key, reviewer_role"
        ),
        "screening_final": (
            "SELECT record_key, final_language, final_decision, "
            "exclusion_reason, h2_required, h2_completed, "
            "adjudication_reason FROM screening_final ORDER BY record_key"
        ),
        "indicator_mentions": ("SELECT * FROM indicator_mentions ORDER BY mention_id"),
        "indicator_source_disposition": (
            "SELECT record_key, disposition, english_fulltext_status, "
            "notes, decided_by FROM indicator_source_disposition "
            "ORDER BY record_key"
        ),
        "indicator_source_reviews": (
            "SELECT record_key, reviewer_role, disposition, "
            "english_fulltext_status, notes FROM indicator_source_reviews "
            "ORDER BY record_key, reviewer_role"
        ),
        "indicator_mention_reviews": (
            "SELECT mention_id, reviewer_role, decision, payload_json, notes "
            "FROM indicator_mention_reviews "
            "ORDER BY mention_id, reviewer_role"
        ),
        "fulltext_acquisitions": (
            "SELECT record_key, candidate_url, final_url, local_path, "
            "sha256, access_statement, http_content_type, byte_count, "
            "status, error FROM fulltext_acquisitions "
            "ORDER BY record_key"
        ),
        "openalex_location_hydration": (
            "SELECT record_key, status, provider_payload_sha256, error "
            "FROM openalex_location_hydration ORDER BY record_key"
        ),
        "indicator_families": ("SELECT * FROM indicator_families ORDER BY feature_id"),
        "feature_data_audit": (
            "SELECT feature_id, data_status, row_count, valid_count, "
            "unique_count, missing_rate, derivation_hash, "
            "input_snapshot_hash, audit_status, reviewer, notes "
            "FROM feature_data_audit ORDER BY feature_id"
        ),
        "dimension_coding": (
            "SELECT feature_id, coder_role, dimension_label, "
            "dimension_definition, construct_role, information_source, "
            "t0_boundary, bias_risk, decision, reason "
            "FROM dimension_coding ORDER BY feature_id, coder_role"
        ),
        "candidate_dimensions": (
            "SELECT * FROM candidate_dimensions ORDER BY dimension_id"
        ),
        "feature_decisions": ("SELECT * FROM feature_decisions ORDER BY feature_id"),
        "dimension_decisions": (
            "SELECT * FROM dimension_decisions ORDER BY dimension_id"
        ),
        "citation_edges": (
            "SELECT * FROM citation_edges ORDER BY source_record_key, "
            "target_provider_id, direction, iteration"
        ),
        "saturation_rounds": (
            "SELECT iteration, new_records, "
            "new_nonredundant_english_terms, "
            "new_canonical_indicator_families, reviewer_role, decision, "
            "notes FROM saturation_rounds ORDER BY iteration"
        ),
        "crossref_validation": (
            "SELECT record_key, doi, status, title_match, year_match, "
            "type_match, crossref_title, crossref_year, crossref_type, "
            "conflict_reason FROM crossref_validation ORDER BY record_key"
        ),
    }
    table_digests: dict[str, dict[str, Any]] = {}
    for name, query in tables.items():
        digest = hashlib.sha256()
        row_count = 0
        for row in connection.execute(query):
            encoded = json.dumps(
                dict(row),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            digest.update(encoded)
            digest.update(b"\n")
            row_count += 1
        table_digests[name] = {
            "rows": row_count,
            "sha256": digest.hexdigest(),
        }
    return {
        "counts": current_counts(connection),
        "table_digests": table_digests,
    }


def _hidden_seed_reconciliation(
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    """Reconcile H2's logged eligible DOI set with imported hidden seeds."""
    logged_dois: set[str] = set()
    invalid_runs: list[str] = []
    for row in connection.execute("""
        SELECT search_run_id, eligible_seed_count,
               eligible_seed_dois_json
        FROM hidden_seed_search_log
        WHERE reviewer_role = 'H2'
          AND completion_status = 'complete'
        ORDER BY search_run_id
        """):
        try:
            raw_dois = json.loads(row["eligible_seed_dois_json"])
        except (json.JSONDecodeError, TypeError):
            invalid_runs.append(str(row["search_run_id"]))
            continue
        if not isinstance(raw_dois, list):
            invalid_runs.append(str(row["search_run_id"]))
            continue
        run_dois = {normalize_doi(value) for value in raw_dois if normalize_doi(value)}
        if len(run_dois) != int(row["eligible_seed_count"]):
            invalid_runs.append(str(row["search_run_id"]))
        logged_dois.update(run_dois)
    seed_dois = {normalize_doi(row["doi"]) for row in connection.execute("""
            SELECT doi FROM evidence_seeds
            WHERE seed_role = 'validation'
              AND supplied_by = 'H2'
              AND hidden_during_development = 1
              AND eligibility_status = 'eligible'
              AND language = 'en'
            """) if normalize_doi(row["doi"])}
    return {
        "logged_dois": sorted(logged_dois),
        "seed_dois": sorted(seed_dois),
        "logged_not_imported": sorted(logged_dois - seed_dois),
        "imported_not_logged": sorted(seed_dois - logged_dois),
        "invalid_runs": sorted(set(invalid_runs)),
        "matches": (not invalid_runs and logged_dois == seed_dois),
    }


def _validated_discovery_phase_freeze(
    connection: sqlite3.Connection,
    phase: str,
    required_zero_rounds: int,
) -> dict[str, Any] | None:
    """Return the latest terminal round when its stop basis verifies."""
    row = connection.execute(
        """
        SELECT * FROM discovery_review_rounds
        WHERE saturation_phase = ?
          AND reviewer_role = 'H2'
          AND decision = 'freeze'
          AND fully_reviewed = 1
          AND iteration = (
              SELECT MAX(iteration)
              FROM discovery_review_rounds
              WHERE saturation_phase = ?
          )
        ORDER BY iteration DESC LIMIT 1
        """,
        (phase, phase),
    ).fetchone()
    if row is None:
        return None
    frozen = dict(row)
    preregistered = (
        int(frozen["new_nonredundant_english_terms"]) == 0
        and int(frozen["new_canonical_indicator_families"]) == 0
        and int(frozen["consecutive_zero_rounds"]) >= required_zero_rounds
        and frozen["stop_basis"] == "preregistered_consecutive_dual_zero"
    )
    if preregistered:
        return frozen
    deviation = (
        phase == "search_frame_discovery"
        and frozen["stop_basis"] == "retrospective_owner_pragmatic_stop"
        and frozen["protocol_amendment_id"]
        == "aspr-v3-search-frame-round12-pragmatic-stop-20260730"
        and DISCOVERY_STOP_AMENDMENT_PATH.exists()
        and frozen["protocol_amendment_sha256"]
        == sha256_file(DISCOVERY_STOP_AMENDMENT_PATH)
    )
    return frozen if deviation else None


def _source_snapshot_blockers(
    connection: sqlite3.Connection,
) -> list[str]:
    """Validate frozen sources and explicitly versioned replacements."""
    blockers: list[str] = []
    supersessions = {str(row["old_source_id"]): row for row in connection.execute("""
            SELECT x.*, n.path AS new_path, n.sha256 AS new_sha256,
                   a.path AS authorization_path,
                   a.sha256 AS authorization_sha256
            FROM source_snapshot_supersessions x
            JOIN source_snapshots n
              ON n.source_id = x.new_source_id
            JOIN source_snapshots a
              ON a.source_id = x.authorization_source_id
            """)}
    for source in connection.execute(
        "SELECT source_id, path, sha256 FROM source_snapshots"
    ):
        path = Path(str(source["path"]))
        if not path.exists():
            blockers.append(f"SOURCE_MISSING:{source['source_id']}")
            continue
        observed = sha256_file(path)
        if observed == source["sha256"]:
            continue
        supersession = supersessions.get(str(source["source_id"]))
        valid_supersession = False
        if supersession is not None:
            new_path = Path(str(supersession["new_path"]))
            authorization_path = Path(str(supersession["authorization_path"]))
            valid_supersession = (
                str(supersession["old_sha256"]) == str(source["sha256"])
                and observed
                == str(supersession["observed_current_sha256"])
                == str(supersession["new_sha256"])
                and new_path.is_file()
                and sha256_file(new_path) == str(supersession["new_sha256"])
                and authorization_path.is_file()
                and sha256_file(authorization_path)
                == str(supersession["authorization_sha256"])
            )
        if not valid_supersession:
            blockers.append(f"SOURCE_HASH_CHANGED:{source['source_id']}")
    return blockers


def _final_training_feature_status(
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    """Verify that the final role-separated decisions have a clean matrix."""
    selected = [str(row["feature_id"]) for row in connection.execute("""
            SELECT d.feature_id
            FROM feature_decisions d
            JOIN indicator_families f USING(feature_id)
            WHERE d.final_role != 'excluded'
            ORDER BY CASE d.final_role
                       WHEN 'predictive' THEN 0
                       WHEN 'opportunity' THEN 1
                       WHEN 'control' THEN 2
                       ELSE 3
                     END,
                     f.feature_id
            """)]
    result: dict[str, Any] = {
        "required": bool(selected),
        "valid": not selected,
        "issues": [],
        "feature_ids": selected,
        "matrix_path": str(FINAL_TRAINING_MATRIX_PATH.resolve()),
        "schema_path": str(FINAL_TRAINING_SCHEMA_PATH.resolve()),
    }
    if not selected:
        return result
    if not FINAL_TRAINING_SCHEMA_PATH.is_file():
        result["issues"].append("SCHEMA_MISSING")
        return result
    if not FINAL_TRAINING_MATRIX_PATH.is_file():
        result["issues"].append("MATRIX_MISSING")
        return result
    try:
        schema = read_json(FINAL_TRAINING_SCHEMA_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        result["issues"].append(f"SCHEMA_INVALID:{type(error).__name__}")
        return result
    matrix_sha256 = sha256_file(FINAL_TRAINING_MATRIX_PATH)
    schema_sha256 = sha256_file(FINAL_TRAINING_SCHEMA_PATH)
    result.update(
        {
            "matrix_sha256": matrix_sha256,
            "schema_sha256": schema_sha256,
        }
    )
    if schema.get("matrix_sha256") != matrix_sha256:
        result["issues"].append("MATRIX_HASH_MISMATCH")
    if list(schema.get("feature_ids") or []) != selected:
        result["issues"].append("FEATURE_DECISION_MISMATCH")
    if int(schema.get("feature_count") or -1) != len(selected):
        result["issues"].append("FEATURE_COUNT_MISMATCH")
    if schema.get("contains_outcomes") is not False:
        result["issues"].append("OUTCOME_COLUMN_NOT_EXCLUDED")
    if schema.get("uses_future_information") is not False:
        result["issues"].append("FUTURE_INFORMATION_NOT_EXCLUDED")
    try:
        frame = pd.read_parquet(FINAL_TRAINING_MATRIX_PATH)
    except (OSError, ValueError, ImportError) as error:
        result["issues"].append(f"MATRIX_INVALID:{type(error).__name__}")
        return result
    if list(frame.columns) != ["paper_id", *selected]:
        result["issues"].append("MATRIX_COLUMN_MISMATCH")
    if (
        "paper_id" not in frame
        or frame["paper_id"].isna().any()
        or frame["paper_id"].duplicated().any()
    ):
        result["issues"].append("INVALID_PAPER_ID")
    if int(schema.get("row_count") or -1) != len(frame):
        result["issues"].append("ROW_COUNT_MISMATCH")
    for feature_id in selected:
        if feature_id not in frame:
            continue
        values = frame[feature_id].dropna()
        if values.empty or int(values.nunique()) <= 1:
            result["issues"].append(f"EMPTY_OR_CONSTANT_FEATURE:{feature_id}")
    result["row_count"] = len(frame)
    result["valid"] = not result["issues"]
    return result


def _completion_blockers(
    connection: sqlite3.Connection,
    stages: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    required_stages = (
        "initialized",
        "bootstrap_retrieval_complete",
        "terms_coded",
        "search_frame_derived",
        "search_frame_validated",
        "search_frame_frozen",
        "formal_retrieval_complete",
        "literature_screened",
        "indicators_extracted",
        "data_correspondence_reviewed",
        "dimensions_derived",
        "features_selected",
    )
    for stage in required_stages:
        if stages.get(stage, {}).get("status") != "complete":
            blockers.append(f"STAGE_INCOMPLETE:{stage}")
    operationalization_amendment = connection.execute(
        """
        SELECT 1 FROM source_snapshots
        WHERE source_id =
              'protocol_amendment_formula_operationalization_separation_v3'
          AND sha256 = ?
        """,
        (sha256_file(FORMULA_OPERATIONALIZATION_AMENDMENT_PATH),),
    ).fetchone()
    eligible_formula_families = connection.execute("""
        SELECT COUNT(*) FROM indicator_families
        WHERE english_fulltext_verified = 1 AND h2_approved = 1
        """).fetchone()[0]
    if (
        operationalization_amendment is not None
        and eligible_formula_families
        and stages.get("operationalizations_reviewed", {}).get("status") != "complete"
    ):
        blockers.append("STAGE_INCOMPLETE:operationalizations_reviewed")
    incomplete_discovery = connection.execute("""
        SELECT COUNT(*)
        FROM discovery_queries q
        LEFT JOIN discovery_query_runs r USING(discovery_query_id)
        WHERE q.status = 'active'
          AND q.query_role != 'formal_search_family'
          AND (r.complete IS NULL OR r.complete != 1)
        """).fetchone()[0]
    if incomplete_discovery:
        blockers.append(f"INCOMPLETE_DISCOVERY_SAMPLE_STRATA:{incomplete_discovery}")
    saturation_protocol = read_json(ROOT / "saturation_protocol_v3.json")
    required_zero_rounds = int(
        saturation_protocol["sequential_review"][
            "minimum_consecutive_zero_novelty_rounds"
        ]
    )
    terminal_amendment = connection.execute(
        """
        SELECT 1 FROM source_snapshots
        WHERE source_id =
              'protocol_amendment_round12_terminal_formal_cohort_v3'
          AND sha256 = ?
        """,
        (sha256_file(TERMINAL_FORMAL_COHORT_AMENDMENT_PATH),),
    ).fetchone()
    terminal_round = connection.execute("""
        SELECT 1 FROM discovery_review_rounds
        WHERE iteration = 12
          AND decision = 'freeze'
          AND fully_reviewed = 1
          AND stop_basis = 'retrospective_owner_pragmatic_stop'
        """).fetchone()
    for phase, blocker_label in (
        (
            "search_frame_discovery",
            "NO_H2_APPROVED_SEARCH_FRAME_DISCOVERY_SATURATION",
        ),
        (
            "formal_indicator_discovery",
            "NO_H2_APPROVED_FORMAL_INDICATOR_SATURATION",
        ),
    ):
        discovery_freeze = _validated_discovery_phase_freeze(
            connection,
            phase,
            required_zero_rounds,
        )
        if discovery_freeze is None:
            if (
                phase == "formal_indicator_discovery"
                and terminal_amendment is not None
                and terminal_round is not None
            ):
                continue
            blockers.append(
                f"{blocker_label}:REQUIRES_{required_zero_rounds}_"
                "CONSECUTIVE_DUAL_ZERO_ROUNDS"
            )
    unresolved_discovery_rounds = connection.execute("""
        SELECT COUNT(*) FROM discovery_review_rounds
        WHERE assigned_records > 0
          AND (
              fully_reviewed != 1
              OR reviewer_role != 'H2'
              OR decision = 'pending'
          )
        """).fetchone()[0]
    if unresolved_discovery_rounds:
        blockers.append(
            f"UNRESOLVED_DISCOVERY_REVIEW_ROUNDS:" f"{unresolved_discovery_rounds}"
        )
    missing_extraction_reviews = connection.execute("""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT h.record_key, h.review_round
            FROM discovery_hits h
            JOIN discovery_review_rounds rr
              ON rr.iteration = h.review_round
            WHERE h.review_status = 'include'
              AND rr.fully_reviewed = 1
              AND NOT EXISTS (
                  SELECT 1 FROM discovery_extraction_reviews e
                  WHERE e.record_key = h.record_key
                    AND e.review_round = h.review_round
                    AND e.reviewer_role = 'H1'
                    AND e.extraction_complete = 1
              )
        )
        """).fetchone()[0]
    if missing_extraction_reviews:
        blockers.append(
            "INCLUDED_DISCOVERY_RECORDS_WITHOUT_COMPLETE_H1_EXTRACTION:"
            f"{missing_extraction_reviews}"
        )
    pending_discovery_indicators = connection.execute("""
        SELECT COUNT(*) FROM discovery_indicator_candidates
        WHERE status = 'candidate'
          AND (
              h1_decision = 'pending'
              OR (h1_decision = 'include' AND h2_decision = 'pending')
          )
        """).fetchone()[0]
    if pending_discovery_indicators:
        blockers.append(
            "UNADJUDICATED_DISCOVERY_INDICATOR_CANDIDATES:"
            f"{pending_discovery_indicators}"
        )
    hidden_seed_count = connection.execute("""
        SELECT COUNT(*) FROM evidence_seeds
        WHERE seed_role = 'validation'
          AND hidden_during_development = 1
          AND supplied_by = 'H2'
          AND eligibility_status = 'eligible'
          AND language = 'en'
        """).fetchone()[0]
    if hidden_seed_count == 0:
        blockers.append("NO_H2_SUPPLIED_HIDDEN_VALIDATION_SEED")
    hidden_seed_routes = {str(row["route"]) for row in connection.execute("""
            SELECT DISTINCT route FROM hidden_seed_search_log
            WHERE reviewer_role = 'H2'
              AND completion_status = 'complete'
            """)}
    required_hidden_seed_routes = {
        "independent_review_search",
        "backward_citation_tracking",
        "forward_citation_tracking",
    }
    missing_hidden_seed_routes = sorted(
        required_hidden_seed_routes - hidden_seed_routes
    )
    if missing_hidden_seed_routes:
        blockers.append(
            "INCOMPLETE_H2_HIDDEN_SEED_SEARCH_PROVENANCE:"
            + ",".join(missing_hidden_seed_routes)
        )
    hidden_seed_reconciliation = _hidden_seed_reconciliation(connection)
    if not hidden_seed_reconciliation["matches"]:
        blockers.append(
            "HIDDEN_SEED_LOG_SEED_SET_MISMATCH:"
            f"invalid_runs={hidden_seed_reconciliation['invalid_runs']};"
            "logged_not_imported="
            f"{hidden_seed_reconciliation['logged_not_imported']};"
            "imported_not_logged="
            f"{hidden_seed_reconciliation['imported_not_logged']}"
        )
    incomplete_queries = connection.execute("""
        SELECT COUNT(*)
        FROM physical_queries p
        JOIN logical_queries l USING(logical_query_id)
        LEFT JOIN query_runs r
          ON r.provider = 'OpenAlex'
         AND r.physical_query_id = p.physical_query_id
         AND r.run_role = 'formal'
        WHERE p.status = 'active' AND l.status = 'active'
          AND l.logical_query_id LIKE 'L%'
          AND (r.complete IS NULL OR r.complete != 1)
        """).fetchone()[0]
    if incomplete_queries:
        blockers.append(f"INCOMPLETE_FORMAL_PHYSICAL_QUERIES:{incomplete_queries}")
    incomplete_citation_queries = connection.execute("""
        SELECT COUNT(*)
        FROM physical_queries p
        JOIN logical_queries l USING(logical_query_id)
        WHERE p.status = 'active' AND l.status = 'citation'
          AND NOT EXISTS (
              SELECT 1 FROM query_runs r
              WHERE r.provider = 'OpenAlex'
                AND r.physical_query_id = p.physical_query_id
                AND r.run_role LIKE 'citation_tracking_%'
                AND r.complete = 1
          )
        """).fetchone()[0]
    if incomplete_citation_queries:
        blockers.append(
            "INCOMPLETE_CITATION_PHYSICAL_QUERIES:" f"{incomplete_citation_queries}"
        )
    missed_seeds = connection.execute("""
        SELECT COUNT(*) FROM evidence_seeds
        WHERE eligibility_status = 'eligible' AND language = 'en'
          AND recall_status NOT IN ('recalled', 'supplemented')
        """).fetchone()[0]
    if missed_seeds:
        blockers.append(f"UNRESOLVED_SEED_RECALL:{missed_seeds}")
    press_issues = connection.execute("""
        SELECT COUNT(*) FROM logical_queries
        WHERE status = 'active' AND logical_query_id LIKE 'L%'
          AND press_status != 'pass'
        """).fetchone()[0]
    if press_issues:
        blockers.append(f"UNRESOLVED_PRESS:{press_issues}")
    formal_records = connection.execute("""
        SELECT COUNT(*)
        FROM formal_review_records r
        LEFT JOIN screening_final s USING(record_key)
        WHERE s.record_key IS NULL
        """).fetchone()[0]
    if formal_records:
        blockers.append(f"RECORDS_WITHOUT_FINAL_DISPOSITION:{formal_records}")
    invalid_screening_spans = connection.execute("""
        SELECT COUNT(*)
        FROM screening_decisions d
        JOIN records r USING(record_key)
        WHERE EXISTS (
            SELECT 1 FROM formal_review_records f
            WHERE f.record_key = d.record_key
        )
          AND (
            d.evidence_span = ''
            OR d.language_evidence = ''
            OR instr(
                r.title || char(10) || r.abstract,
                d.evidence_span
            ) = 0
            OR instr(
                r.title || char(10) || r.abstract,
                d.language_evidence
            ) = 0
          )
        """).fetchone()[0]
    if invalid_screening_spans:
        blockers.append(
            "SCREENING_DECISIONS_WITHOUT_EXACT_SOURCE_SPANS:"
            f"{invalid_screening_spans}"
        )
    independent_indicator_review_failures = connection.execute("""
        WITH included AS (
            SELECT record_key FROM screening_final
            WHERE final_decision = 'include' AND final_language = 'en'
        ),
        failures AS (
            SELECT 'source_h1:' || i.record_key AS failure
            FROM included i
            LEFT JOIN indicator_source_reviews r
              ON r.record_key = i.record_key
             AND r.reviewer_role = 'H1'
            WHERE r.record_key IS NULL
            UNION ALL
            SELECT 'source_h2:' || i.record_key
            FROM included i
            LEFT JOIN indicator_source_reviews r
              ON r.record_key = i.record_key
             AND r.reviewer_role = 'H2'
            WHERE r.record_key IS NULL
            UNION ALL
            SELECT 'mention_h1:' || m.mention_id
            FROM indicator_mentions m
            LEFT JOIN indicator_mention_reviews r
              ON r.mention_id = m.mention_id
             AND r.reviewer_role = 'H1'
            WHERE r.mention_id IS NULL
            UNION ALL
            SELECT 'mention_h2:' || h1.mention_id
            FROM indicator_mention_reviews h1
            LEFT JOIN indicator_mention_reviews h2
              ON h2.mention_id = h1.mention_id
             AND h2.reviewer_role = 'H2'
            WHERE h1.reviewer_role = 'H1'
              AND h1.decision != 'excluded'
              AND h2.mention_id IS NULL
        )
        SELECT COUNT(*) FROM failures
        """).fetchone()[0]
    if independent_indicator_review_failures:
        blockers.append(
            "INCOMPLETE_INDEPENDENT_INDICATOR_REVIEWS:"
            f"{independent_indicator_review_failures}"
        )
    selected_formula_failures = connection.execute(
        """
        SELECT COUNT(*)
        FROM indicator_families f
        JOIN feature_decisions d USING(feature_id)
        WHERE d.final_role != 'excluded'
          AND (
            f.english_fulltext_verified != 1
            OR (
                ? = 0
                AND f.formula_reproducible != 1
            )
            OR (
                ? = 1
                AND NOT EXISTS (
                    SELECT 1
                    FROM feature_operationalization_reviews o
                    WHERE o.feature_id = f.feature_id
                      AND o.reviewer_role = 'H2'
                      AND o.decision = 'approve'
                )
            )
            OR f.h2_approved != 1
            OR NOT EXISTS (
                SELECT 1
                FROM json_each(f.mention_ids_json) j
                JOIN indicator_mentions m ON m.mention_id = j.value
                WHERE m.english_fulltext_verified = 1
                  AND m.formula_reproducible = 1
                  AND m.h2_approved = 1
                  AND m.fulltext_source_url != ''
                  AND m.fulltext_local_path != ''
                  AND m.fulltext_sha256 != ''
                  AND m.fulltext_license != ''
            )
          )
        """,
        (
            int(operationalization_amendment is not None),
            int(operationalization_amendment is not None),
        ),
    ).fetchone()[0]
    if selected_formula_failures:
        blockers.append(
            "SELECTED_INDICATORS_WITHOUT_FULLTEXT_FORMULA_OR_H2:"
            f"{selected_formula_failures}"
        )
    failed_selected_gates = connection.execute("""
        SELECT COUNT(*) FROM feature_decisions
        WHERE final_role != 'excluded' AND failed_gates_json != '[]'
        """).fetchone()[0]
    if failed_selected_gates:
        blockers.append(f"SELECTED_INDICATORS_FAILED_GATES:{failed_selected_gates}")
    missing_selected_data_artifacts = connection.execute("""
        SELECT COUNT(*)
        FROM feature_decisions d
        LEFT JOIN feature_data_audit a USING(feature_id)
        WHERE d.final_role != 'excluded'
          AND (
            a.feature_id IS NULL
            OR a.derivation_artifact_path = ''
            OR a.input_snapshot_path = ''
            OR a.derivation_hash = ''
            OR a.input_snapshot_hash = ''
            OR NOT EXISTS (
                SELECT 1 FROM source_snapshots s
                WHERE s.source_id =
                        'feature_derivation_artifact_' || d.feature_id
                  AND s.sha256 = a.derivation_hash
            )
            OR NOT EXISTS (
                SELECT 1 FROM source_snapshots s
                WHERE s.source_id =
                        'feature_input_snapshot_' || d.feature_id
                  AND s.sha256 = a.input_snapshot_hash
            )
          )
        """).fetchone()[0]
    if missing_selected_data_artifacts:
        blockers.append(
            "SELECTED_INDICATORS_WITHOUT_VERIFIED_DATA_ARTIFACTS:"
            f"{missing_selected_data_artifacts}"
        )
    invalid_dimensions = connection.execute("""
        SELECT COUNT(*) FROM dimension_decisions
        WHERE selected = 1
          AND (
            independent_group_count < 2
            OR selected_feature_ids_json = '[]'
          )
        """).fetchone()[0]
    if invalid_dimensions:
        blockers.append(f"INVALID_RETAINED_DIMENSIONS:{invalid_dimensions}")
    saturation = connection.execute("""
        SELECT * FROM saturation_rounds
        WHERE reviewer_role = 'H2' AND decision = 'freeze'
          AND new_nonredundant_english_terms = 0
          AND new_canonical_indicator_families = 0
        ORDER BY iteration DESC LIMIT 1
        """).fetchone()
    if saturation is None and (terminal_amendment is None or terminal_round is None):
        blockers.append("NO_ZERO_NOVELTY_OR_REGISTERED_ROUND12_TERMINAL_AMENDMENT")
    conflicts = connection.execute("""
        SELECT COUNT(*)
        FROM formal_review_records r
        JOIN crossref_validation c USING(record_key)
        WHERE c.status IN ('conflict', 'error')
        """).fetchone()[0]
    if conflicts:
        blockers.append(f"UNRESOLVED_CROSSREF_CONFLICTS:{conflicts}")
    unvalidated_dois = connection.execute("""
        SELECT COUNT(*)
        FROM formal_review_records s
        JOIN records r USING(record_key)
        LEFT JOIN crossref_validation c USING(record_key)
        WHERE r.doi != '' AND c.record_key IS NULL
        """).fetchone()[0]
    if unvalidated_dois:
        blockers.append(f"FORMAL_DOI_RECORDS_NOT_CROSSREF_VALIDATED:{unvalidated_dois}")
    blockers.extend(_source_snapshot_blockers(connection))
    training_status = _final_training_feature_status(connection)
    if training_status["required"] and not training_status["valid"]:
        blockers.extend(
            f"FINAL_TRAINING_FEATURES:{issue}" for issue in training_status["issues"]
        )
    return blockers


def _completion_matrix(
    connection: sqlite3.Connection,
) -> list[dict[str, Any]]:
    """Map the full objective to direct, reader-auditable evidence."""
    stages = _stage_summary(connection)
    assigned = int(connection.execute("""
            SELECT COUNT(*) FROM formal_review_records
            """).fetchone()[0])
    screen_counts = {
        role: int(
            connection.execute(
                """
                SELECT COUNT(DISTINCT d.record_key)
                FROM screening_decisions d
                WHERE d.reviewer_role = ?
                  AND EXISTS (
                      SELECT 1 FROM formal_review_records f
                      WHERE f.record_key = d.record_key
                  )
                """,
                (role,),
            ).fetchone()[0]
        )
        for role in ("AI", "H1", "H2")
    }
    invalid_primary_screening_spans = int(connection.execute("""
            SELECT COUNT(*)
            FROM screening_decisions d
            JOIN records r USING(record_key)
            WHERE d.reviewer_role IN ('AI', 'H1')
              AND EXISTS (
                  SELECT 1 FROM formal_review_records f
                  WHERE f.record_key = d.record_key
              )
              AND (
                    d.language_evidence = ''
                    OR d.evidence_span = ''
                    OR instr(
                        r.title || char(10) || r.abstract,
                        d.language_evidence
                    ) = 0
                    OR instr(
                        r.title || char(10) || r.abstract,
                        d.evidence_span
                    ) = 0
              )
            """).fetchone()[0])
    active_terms = int(
        connection.execute(
            "SELECT COUNT(*) FROM raw_terms WHERE status = 'active'"
        ).fetchone()[0]
    )
    term_counts = {
        role: int(
            connection.execute(
                """
                SELECT COUNT(*) FROM term_coding
                WHERE coder_role = ?
                  AND term_id IN (
                      SELECT term_id FROM raw_terms WHERE status = 'active'
                  )
                """,
                (role,),
            ).fetchone()[0]
        )
        for role in ("AI", "H1", "H2")
    }
    term_h2_required = 0
    term_h2_complete = 0
    for term in connection.execute(
        "SELECT term_id FROM raw_terms WHERE status = 'active'"
    ):
        codes = {
            str(row["coder_role"]): row
            for row in connection.execute(
                """
                SELECT * FROM term_coding
                WHERE term_id = ?
                """,
                (term["term_id"],),
            )
        }
        if "AI" not in codes or "H1" not in codes:
            continue

        def signature(row: sqlite3.Row) -> tuple[Any, ...]:
            return (
                row["decision"],
                normalize_term(row["canonical_term"]),
                normalize_term(row["term_family_label"]),
                row["term_relation"],
                normalize_term(row["search_domain_label"]),
                normalize_term(row["search_domain_definition"]),
                normalize_term(row["query_family_label"]),
                int(row["cross_domain"]),
            )

        concordant_exclusion = codes["AI"]["decision"] == "exclude" and signature(
            codes["AI"]
        ) == signature(codes["H1"])
        if not concordant_exclusion:
            term_h2_required += 1
            term_h2_complete += int("H2" in codes)
    strata = connection.execute("""
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN r.complete = 1 THEN 1 ELSE 0 END) AS complete
        FROM discovery_queries q
        LEFT JOIN discovery_query_runs r USING(discovery_query_id)
        WHERE q.status = 'active'
          AND q.query_role NOT IN (
              'development_citation_network',
              'citation_tracking_network',
              'formal_search_family'
          )
        """).fetchone()
    fully_reviewed_rounds = int(connection.execute("""
            SELECT COUNT(*) FROM discovery_review_rounds
            WHERE fully_reviewed = 1
            """).fetchone()[0])
    total_rounds = int(
        connection.execute("SELECT COUNT(*) FROM discovery_review_rounds").fetchone()[0]
    )
    required_zero_rounds = int(
        read_json(ROOT / "saturation_protocol_v3.json")["sequential_review"][
            "minimum_consecutive_zero_novelty_rounds"
        ]
    )
    phase_freeze_rows = {
        phase: _validated_discovery_phase_freeze(
            connection,
            phase,
            required_zero_rounds,
        )
        for phase in (
            "search_frame_discovery",
            "formal_indicator_discovery",
        )
    }
    phase_freezes = {
        phase: int(row is not None) for phase, row in phase_freeze_rows.items()
    }
    terminal_formal_cohort_authorized = bool(
        connection.execute(
            """
            SELECT 1 FROM source_snapshots
            WHERE source_id =
                  'protocol_amendment_round12_terminal_formal_cohort_v3'
              AND sha256 = ?
            """,
            (sha256_file(TERMINAL_FORMAL_COHORT_AMENDMENT_PATH),),
        ).fetchone()
        and connection.execute("""
            SELECT 1 FROM discovery_review_rounds
            WHERE iteration = 12
              AND decision = 'freeze'
              AND fully_reviewed = 1
              AND stop_basis = 'retrospective_owner_pragmatic_stop'
            """).fetchone()
    )
    hidden_seeds = int(connection.execute("""
            SELECT COUNT(*) FROM evidence_seeds
            WHERE seed_role = 'validation'
              AND supplied_by = 'H2'
              AND hidden_during_development = 1
              AND eligibility_status = 'eligible'
              AND language = 'en'
            """).fetchone()[0])
    complete_hidden_routes = int(connection.execute("""
            SELECT COUNT(DISTINCT route) FROM hidden_seed_search_log
            WHERE reviewer_role = 'H2'
              AND completion_status = 'complete'
            """).fetchone()[0])
    hidden_seed_reconciliation = _hidden_seed_reconciliation(connection)
    counts = current_counts(connection)
    rows: list[dict[str, Any]] = []

    def add(
        requirement_id: str,
        phase: str,
        requirement: str,
        passed: bool,
        evidence: str,
        remaining_action: str,
        authority: str,
    ) -> None:
        rows.append(
            {
                "requirement_id": requirement_id,
                "phase": phase,
                "requirement": requirement,
                "status": "PASS" if passed else "PENDING",
                "evidence": evidence,
                "remaining_action": "" if passed else remaining_action,
                "decision_authority": authority,
            }
        )

    add(
        "R01",
        "scope",
        "Standalone v3 with frozen English-only T0 protocol and no quotas",
        stages.get("initialized", {}).get("status") == "complete",
        "protocol_v3.json; source_snapshots; initialized stage",
        "Initialize and verify all frozen source hashes",
        "protocol",
    )
    add(
        "R02",
        "bootstrap",
        "Domain-agnostic OpenAlex inventory completed",
        stages.get("bootstrap_inventory_complete", {}).get("status") == "complete",
        "bootstrap_inventory_complete stage; reported total 3,591,214",
        "Complete bootstrap inventory",
        "system",
    )
    add(
        "R03",
        "discovery",
        "Every active deterministic discovery stratum is retrieved",
        int(strata["total"] or 0) > 0
        and int(strata["complete"] or 0) == int(strata["total"] or 0),
        (
            f"complete strata={int(strata['complete'] or 0)}/"
            f"{int(strata['total'] or 0)}"
        ),
        "Resume incomplete cursor/sample strata",
        "system",
    )
    add(
        "R04",
        "screening",
        "AI and H1 each screen every assigned record with frozen provenance",
        assigned > 0
        and screen_counts["AI"] == assigned
        and screen_counts["H1"] == assigned
        and invalid_primary_screening_spans == 0,
        (
            f"assigned={assigned}; AI={screen_counts['AI']}; "
            f"H1={screen_counts['H1']}; invalid exact spans="
            f"{invalid_primary_screening_spans}"
        ),
        "Normalize legacy AI provenance, then import H1 decisions or an "
        "exact human-attested reviewed draft",
        "AI + H1",
    )
    add(
        "R05",
        "screening",
        "Every discovery round is adjudicated and fully reviewed",
        total_rounds > 0 and fully_reviewed_rounds == total_rounds,
        (
            f"fully reviewed rounds={fully_reviewed_rounds}/"
            f"{total_rounds}; H2 decisions={screen_counts['H2']}"
        ),
        "Generate/import mandatory H2 queues and finalize each round",
        "H2",
    )
    add(
        "R06",
        "term coding",
        "Every active term has role-separated AI/H1 coding and H2 disposition",
        active_terms > 0
        and term_counts["AI"] == active_terms
        and term_counts["H1"] == active_terms
        and term_h2_complete == term_h2_required,
        (
            f"active terms={active_terms}; AI={term_counts['AI']}; "
            f"H1={term_counts['H1']}; mandatory H2="
            + (
                f"{term_h2_complete}/{term_h2_required}"
                if term_counts["AI"] == active_terms
                and term_counts["H1"] == active_terms
                else "not computable until AI/H1 are complete"
            )
        ),
        "Complete H1 term coding with provenance and H2 adjudication",
        "AI + H1 + H2",
    )
    add(
        "R07",
        "search-frame saturation",
        (
            "Verified terminal search-frame discovery decision "
            "(preregistered dual-zero or disclosed protocol deviation)"
        ),
        phase_freezes["search_frame_discovery"] > 0,
        (
            "qualifying freezes="
            f"{phase_freezes['search_frame_discovery']}; stop basis="
            + (
                str(phase_freeze_rows["search_frame_discovery"]["stop_basis"])
                if phase_freeze_rows["search_frame_discovery"]
                else "none"
            )
        ),
        (
            "Complete the terminal round and disclose any departure from "
            "the preregistered three-round dual-zero rule"
        ),
        "H2",
    )
    add(
        "R08",
        "external validation",
        "H2 documents and adopts hidden-seed search and citation routes",
        complete_hidden_routes >= 3
        and hidden_seeds > 0
        and hidden_seed_reconciliation["matches"],
        (
            f"complete route classes={complete_hidden_routes}/3; "
            f"eligible English hidden seeds={hidden_seeds}; "
            "logged/imported DOI sets match="
            f"{hidden_seed_reconciliation['matches']}"
        ),
        "H2 imports three provenance routes and all eligible hidden seeds",
        "H2",
    )
    add(
        "R09",
        "search frame",
        "Evidence-derived K/Q/P are derived, PRESS-reviewed, and recalled",
        stages.get("search_frame_validated", {}).get("status") == "complete",
        f"K={counts['K']}; Q={counts['Q']}; P={counts['P']}",
        "Derive frame, complete PRESS, and recall every eligible seed",
        "system + H2",
    )
    add(
        "R10",
        "search frame",
        "Validated K/Q/P search frame is frozen with a hash",
        stages.get("search_frame_frozen", {}).get("status") == "complete",
        "search_frame_frozen stage and frozen_search_frame_v3.json",
        "Freeze the validated search frame",
        "system",
    )
    add(
        "R11",
        "formal retrieval",
        "Frozen formal pools and citation routes reach independent saturation",
        stages.get("formal_retrieval_complete", {}).get("status") == "complete"
        and (
            phase_freezes["formal_indicator_discovery"] > 0
            or terminal_formal_cohort_authorized
        ),
        (
            "formal retrieval="
            f"{stages.get('formal_retrieval_complete', {}).get('status')}; "
            "formal qualifying freezes="
            f"{phase_freezes['formal_indicator_discovery']}; "
            "round12 terminal fixed-cohort amendment="
            f"{terminal_formal_cohort_authorized}"
        ),
        (
            "Retrieve and review the registered fixed formal cohort, or "
            "complete the preregistered formal dual-zero endpoint"
        ),
        "system + H2",
    )
    add(
        "R12",
        "literature",
        "Every reviewed formal record has a final disposition",
        stages.get("literature_screened", {}).get("status") == "complete",
        (
            "literature_screened stage="
            f"{stages.get('literature_screened', {}).get('status')}"
        ),
        "Complete AI/H1 screening and all H2 adjudication",
        "AI + H1 + H2",
    )
    add(
        "R13",
        "indicators",
        "Every final indicator has English full-text formula evidence",
        stages.get("indicators_extracted", {}).get("status") == "complete"
        and counts["F"] > 0,
        (
            "indicators_extracted stage="
            f"{stages.get('indicators_extracted', {}).get('status')}; "
            f"F={counts['F']}"
        ),
        "Extract and verify full-text formulas and provenance",
        "H1 + H2",
    )
    add(
        "R14",
        "dimensions",
        "M is derived after canonical indicator-family coding",
        stages.get("dimensions_derived", {}).get("status") == "complete",
        f"M={counts['M']}",
        "Complete AI/H1 construct coding and H2 merge/split adjudication",
        "AI + H1 + H2",
    )
    add(
        "R15",
        "selection",
        "Fixed hard gates and redundancy rules determine D and F",
        stages.get("features_selected", {}).get("status") == "complete",
        f"D={counts['D']}; F={counts['F']}",
        "Complete data audits and execute deterministic gate selection",
        "system + H2",
    )
    add(
        "R16",
        "audit",
        "No blocker remains and frozen-input reruns reproduce the result",
        stages.get("audit_complete", {}).get("status") == "complete",
        ("audit_complete stage=" f"{stages.get('audit_complete', {}).get('status')}"),
        "Resolve every audit blocker and repeat the final audit",
        "system",
    )
    return rows


def _export_evidence_tables(connection: sqlite3.Connection) -> None:
    exports: Sequence[tuple[str, str]] = (
        (
            "discovery_queries_v3.csv",
            "SELECT * FROM discovery_queries ORDER BY discovery_query_id",
        ),
        (
            "discovery_query_evidence_v3.csv",
            "SELECT * FROM discovery_query_evidence " "ORDER BY discovery_query_id",
        ),
        (
            "discovery_query_runs_v3.csv",
            "SELECT * FROM discovery_query_runs " "ORDER BY discovery_query_id",
        ),
        (
            "discovery_review_rounds_v3.csv",
            "SELECT * FROM discovery_review_rounds ORDER BY iteration",
        ),
        (
            "saturation_curve_v3.csv",
            "SELECT iteration, saturation_phase, assigned_records, "
            "fully_reviewed, "
            "new_nonredundant_english_terms, "
            "new_canonical_indicator_families, "
            "SUM(CASE WHEN new_nonredundant_english_terms >= 0 "
            "THEN new_nonredundant_english_terms ELSE 0 END) "
            "OVER (ORDER BY iteration) AS cumulative_new_terms, "
            "SUM(CASE WHEN new_canonical_indicator_families >= 0 "
            "THEN new_canonical_indicator_families ELSE 0 END) "
            "OVER (ORDER BY iteration) "
            "AS cumulative_new_indicator_families, "
            "consecutive_zero_rounds, reviewer_role, decision, "
            "stop_basis, protocol_amendment_id, "
            "protocol_amendment_sha256, notes "
            "FROM discovery_review_rounds ORDER BY iteration",
        ),
        (
            "discovery_indicator_candidates_v3.csv",
            "SELECT * FROM discovery_indicator_candidates " "ORDER BY candidate_id",
        ),
        (
            "discovery_extraction_reviews_v3.csv",
            "SELECT * FROM discovery_extraction_reviews "
            "ORDER BY review_round, record_key, reviewer_role",
        ),
        (
            "api_budget_observations_v3.csv",
            "SELECT * FROM api_budget_observations " "ORDER BY observation_id",
        ),
        (
            "open_fulltext_acquisitions_v3.csv",
            "SELECT * FROM fulltext_acquisitions ORDER BY record_key",
        ),
        (
            "openalex_location_hydration_v3.csv",
            "SELECT * FROM openalex_location_hydration " "ORDER BY record_key",
        ),
        (
            "local_snapshot_sources_v3.csv",
            "SELECT * FROM local_snapshot_sources ORDER BY snapshot_id",
        ),
        (
            "ai_assistance_runs_v3.csv",
            "SELECT * FROM ai_assistance_runs ORDER BY started_at, run_id",
        ),
        (
            "human_review_attestations_v3.csv",
            "SELECT * FROM human_review_attestations " "ORDER BY attestation_id",
        ),
        (
            "independent_ai_review_runs_v3.csv",
            "SELECT * FROM independent_ai_review_runs ORDER BY run_id",
        ),
        (
            "review_run_supersessions_v3.csv",
            "SELECT * FROM review_run_supersessions ORDER BY old_run_id",
        ),
        (
            "english_raw_terms_v3.csv",
            "SELECT * FROM raw_terms ORDER BY term_id",
        ),
        (
            "term_coding_v3.csv",
            "SELECT * FROM term_coding ORDER BY term_id, coder_role",
        ),
        (
            "canonical_terms_v3.csv",
            "SELECT * FROM canonical_terms ORDER BY canonical_term_id",
        ),
        (
            "term_families_v3.csv",
            "SELECT * FROM term_families ORDER BY term_family_id",
        ),
        (
            "search_domains_v3.csv",
            "SELECT * FROM search_domains ORDER BY search_domain_id",
        ),
        (
            "search_frame_versions_v3.csv",
            "SELECT * FROM search_frame_versions ORDER BY frame_version",
        ),
        (
            "logical_queries_v3.csv",
            "SELECT * FROM logical_queries WHERE logical_query_id LIKE 'L%' "
            "ORDER BY logical_query_id",
        ),
        (
            "physical_queries_v3.csv",
            "SELECT * FROM physical_queries "
            "WHERE logical_query_id LIKE 'L%' ORDER BY physical_query_id",
        ),
        (
            "query_retrieval_runs_v3.csv",
            "SELECT * FROM query_runs "
            "ORDER BY provider, physical_query_id, run_role",
        ),
        (
            "press_review_v3.csv",
            "SELECT * FROM press_reviews ORDER BY logical_query_id",
        ),
        (
            "seed_recall_v3.csv",
            "SELECT * FROM evidence_seeds ORDER BY seed_role, seed_id",
        ),
        (
            "hidden_seed_search_log_v3.csv",
            "SELECT * FROM hidden_seed_search_log " "ORDER BY search_run_id",
        ),
        (
            "literature_dispositions_v3.csv",
            "SELECT r.record_key, r.doi, r.title, r.language AS "
            "openalex_language, s.* FROM records r "
            "JOIN screening_final s USING(record_key) ORDER BY r.record_key",
        ),
        (
            "fixed_formal_screening_scope_v3.csv",
            "SELECT f.record_key, r.doi, r.title, r.language AS "
            "openalex_language, "
            "EXISTS(SELECT 1 FROM discovery_hits d "
            "WHERE d.record_key=f.record_key AND d.review_round>0) "
            "AS reviewed_in_rounds_1_12, "
            "COALESCE(MIN(CASE WHEN q.run_role='formal' "
            "THEN q.rank END), 0) AS best_formal_seeded_rank, "
            "GROUP_CONCAT(DISTINCT CASE WHEN q.run_role='formal' "
            "AND q.rank BETWEEN 1 AND 10 "
            "THEN q.physical_query_id END) AS formal_query_ids "
            "FROM formal_review_records f "
            "JOIN records r USING(record_key) "
            "LEFT JOIN query_hits q USING(record_key) "
            "GROUP BY f.record_key, r.doi, r.title, r.language "
            "ORDER BY f.record_key",
        ),
        (
            "citation_edges_v3.csv",
            "SELECT * FROM citation_edges "
            "ORDER BY source_record_key, target_provider_id, direction, "
            "iteration",
        ),
        (
            "source_snapshot_supersessions_v3.csv",
            "SELECT * FROM source_snapshot_supersessions " "ORDER BY old_source_id",
        ),
        (
            "crossref_validation_v3.csv",
            "SELECT * FROM crossref_validation ORDER BY record_key",
        ),
        (
            "indicator_mentions_v3.csv",
            "SELECT * FROM indicator_mentions ORDER BY mention_id",
        ),
        (
            "indicator_source_reviews_v3.csv",
            "SELECT * FROM indicator_source_reviews "
            "ORDER BY record_key, reviewer_role",
        ),
        (
            "indicator_mention_reviews_v3.csv",
            "SELECT * FROM indicator_mention_reviews "
            "ORDER BY mention_id, reviewer_role",
        ),
        (
            "complete_indicator_library_v3.csv",
            "SELECT * FROM indicator_families ORDER BY feature_id",
        ),
        (
            "complete_indicator_library.csv",
            "SELECT * FROM indicator_families ORDER BY feature_id",
        ),
        (
            "feature_data_audit_v3.csv",
            "SELECT * FROM feature_data_audit ORDER BY feature_id",
        ),
        (
            "candidate_dimensions_v3.csv",
            "SELECT * FROM candidate_dimensions ORDER BY dimension_id",
        ),
        (
            "candidate_dimensions.csv",
            "SELECT * FROM candidate_dimensions ORDER BY dimension_id",
        ),
        (
            "dimension_merge_split_log_v3.csv",
            "SELECT * FROM dimension_coding " "ORDER BY feature_id, coder_role",
        ),
        (
            "feature_gate_decisions_v3.csv",
            "SELECT * FROM feature_decisions ORDER BY feature_id",
        ),
        (
            "feature_gate_decisions.csv",
            "SELECT * FROM feature_decisions ORDER BY feature_id",
        ),
    )
    for filename, query in exports:
        cursor = connection.execute(query)
        fields = [str(description[0]) for description in (cursor.description or [])]
        write_csv_iter(
            OUTPUT_DIR / filename,
            (dict(row) for row in cursor),
            fields,
        )
    write_csv_iter(
        OUTPUT_DIR / "completion_matrix_v3.csv",
        _completion_matrix(connection),
        (
            "requirement_id",
            "phase",
            "requirement",
            "status",
            "evidence",
            "remaining_action",
            "decision_authority",
        ),
    )


def _write_report(summary: Mapping[str, Any]) -> None:
    counts = summary["counts"]
    prisma = summary["prisma"]
    discovery = summary["discovery"]
    terminal_round = discovery.get("latest_review_round") or {}
    blockers = summary["completion_blockers"]
    invalidated_h1 = summary["invalidated_automated_h1_artifacts"]
    invalidated_qwen = summary["invalidated_local_qwen_review_artifacts"]
    unreviewed_h2 = summary["unreviewed_automated_h2_artifacts"]
    attested_reviews = summary["human_attested_review_artifacts"]
    independent_reviews = summary["independent_ai_reviews"]
    codex_review_artifacts = summary["independent_codex_review_artifacts"]
    retained_dimensions = summary["retained_dimensions"]
    final_indicators = summary["final_indicators"]
    training_features = summary["final_training_features"]
    source_supersessions = summary["source_snapshot_supersessions"]
    status = "COMPLETE" if summary["formal_review_complete"] else "INCOMPLETE"
    bootstrap_inventory = (
        summary["stages"]
        .get("bootstrap_inventory_complete", {})
        .get("details", {})
        .get("reported_total", 0)
    )
    lines = [
        "# Evidence-derived v3 audit report",
        "",
        f"Formal review status: **{status}**",
        "",
        "## Bootstrap inventory",
        "",
        f"- Domain-agnostic OpenAlex records reported: " f"{bootstrap_inventory}",
        "- This is an inventory count, not a retrieved or screened count.",
        "",
        "## Evidence-saturation discovery",
        "",
        f"- Active deterministic sample strata: "
        f"{discovery['active_sample_strata']}",
        f"- Completely retrieved sample strata: "
        f"{discovery['complete_sample_strata']}",
        f"- Within-stratum rows before cross-stratum deduplication: "
        f"{discovery['within_stratum_rows_before_deduplication']}",
        f"- Unique records in deterministic sample strata: "
        f"{discovery['unique_deterministic_sample_records']}",
        f"- Unique records in the development-seed citation network: "
        f"{discovery['unique_development_citation_network_records']}",
        f"- Unique discovery/citation-network records: "
        f"{discovery['unique_discovery_records']}",
        f"- Records assigned to sequential role-separated review: "
        f"{discovery['assigned_review_records']}",
        "- The broad query was not exhaustively downloaded; the defensible "
        "claim is a systematic deterministic evidence map.",
        f"- Terminal discovery round: "
        f"{terminal_round.get('iteration', 'not recorded')}",
        f"- Terminal stop basis: "
        f"{terminal_round.get('stop_basis', 'not recorded')}",
        f"- Terminal actual new term families: "
        f"{terminal_round.get('new_nonredundant_english_terms', 'not recorded')}",
        f"- Terminal actual new indicator families: "
        f"{terminal_round.get('new_canonical_indicator_families', 'not recorded')}",
        "- External post-freeze expansion: 0 new term families / 0 new "
        "indicator families (no round 13 or later saturation round).",
        "- The post-freeze 0/0 shorthand does not relabel the preceding "
        "within-round-12 endpoint values. The pragmatic stop remains a "
        "disclosed retrospective protocol deviation.",
        "",
        "## Emergent counts",
        "",
        f"- K search concept domains: {counts['K']}",
        f"- Q logical query families: {counts['Q']}",
        f"- P physical OpenAlex requests: {counts['P']}",
        f"- M candidate model dimensions: {counts['M']}",
        f"- D retained predictive dimensions: {counts['D']}",
        f"- F final indicators across all retained roles: {counts['F']}",
        "",
        "These values are outputs of coded evidence, PRESS/recall validation, "
        "formula verification, fixed hard gates, deterministic redundancy "
        "resolution, and dimension-retention rules. No numerical quota or "
        "per-dimension allocation was used.",
        "",
        "## Retained dimensions and final indicators",
        "",
    ]
    if retained_dimensions:
        for dimension in retained_dimensions:
            feature_ids = ", ".join(json.loads(dimension["selected_feature_ids_json"]))
            lines.append(
                f"- [{dimension['dimension_role']}] "
                f"{dimension['label']} ({dimension['dimension_id']}): "
                f"{feature_ids}; independent research groups="
                f"{dimension['independent_group_count']}."
            )
    else:
        lines.append("- No dimension has passed all formal retention gates.")
    lines.extend(("", "Final model features:", ""))
    if final_indicators:
        for feature in final_indicators:
            lines.append(
                f"- [{feature['final_role']}] {feature['feature_id']} — "
                f"{feature['canonical_name_en']} / {feature['label_zh']}; "
                f"valid={feature['valid_count']}, "
                f"missing_rate={float(feature['missing_rate']):.4f}, "
                f"unique={feature['unique_count']}."
            )
    else:
        lines.append("- No indicator has passed all formal hard gates.")
    lines.extend(
        (
            "",
            f"Training matrix: {training_features.get('row_count', 0)} rows, "
            f"{len(training_features.get('feature_ids', []))} features; "
            f"SHA-256 "
            f"`{training_features.get('matrix_sha256', 'not available')}`.",
            "",
            "## PRISMA-style flow",
            "",
            f"- Formal OpenAlex query-record links: "
            f"{prisma['openalex_formal_query_record_links']}",
            f"- Unique formal OpenAlex records after query-link deduplication: "
            f"{prisma['openalex_unique_formal_query_records']}",
            f"- Citation-route records: " f"{prisma['unique_citation_route_records']}",
            f"- Records with a final title/abstract disposition: "
            f"{prisma['records_with_final_title_abstract_disposition']}",
            f"- English records included for the indicator census: "
            f"{prisma['english_records_included_for_fulltext_indicator_census']}",
            "",
            "## Language boundary and bias",
            "",
            "Only English publications are eligible. Non-English records are "
            "retained in the retrieval/screening denominator and excluded with "
            "`E_LANGUAGE_NON_ENGLISH`. This restriction may underrepresent "
            "research traditions, constructs, venues, and indicator validation "
            "evidence from non-English-speaking regions; the resulting feature "
            "space therefore carries language and geographic coverage bias.",
            "",
            "## Reviewer provenance and integrity",
            "",
            f"- Human-reviewed automated-draft decision files accepted under "
            f"the project-owner attestation: {len(attested_reviews)}",
            "- Accepted files retain the explicit provenance label "
            "`human_attested_automated_draft`; they are not represented as "
            "originally manual or independently generated blind drafts.",
            "- Agreement involving this H1 batch is concordance with the "
            "adopted reviewed draft, not evidence of independent manual draft "
            "generation.",
            f"- Registered independent-AI review runs: "
            f"{independent_reviews['run_count']}",
            f"- Registered independent-AI reviewed rows: "
            f"{independent_reviews['item_count']}",
            f"- Separate-Codex review CSV/manifest artifacts retained: "
            f"{len(codex_review_artifacts)}",
            "- New reviewer substitutions are explicitly labelled "
            "`independent_ai`; they are never represented as human judgments.",
            "- Reviewer substitution uses a separate Codex task under a frozen "
            "brief; local Ollama/Qwen execution is forbidden. The result "
            "remains AI review rather than human review.",
            "- The H1 draft's fixed 28-label term dictionary remains "
            "provisional; it cannot determine K without H2 merge/split "
            "adjudication, direct source support, PRESS, and seed recall.",
            f"- Invalidated automated H1 trial artifacts retained for "
            f"provenance: {len(invalidated_h1)}",
            "- These files were never imported as H1/H2 decisions and are "
            "excluded from all agreement, saturation, domain, dimension, and "
            "indicator calculations.",
            f"- Invalidated local-Qwen review artifacts retained for "
            f"provenance: {len(invalidated_qwen)}",
            "- The local-Qwen run was stopped at the project owner's request "
            "and is excluded from import, review, agreement, and all final "
            "counts.",
            f"- Unreviewed automated H2 assistance artifacts retained and "
            f"blocked from import: {len(unreviewed_h2)}",
            "- These H2 files are drafts only. Directory and decision-file "
            "hash guards prevent direct use; a reviewed derivative requires "
            "either an exact human attestation or a registered independent-AI "
            "run manifest.",
            "",
            "## Completion blockers",
            "",
        )
    )
    if blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("- None")
    lines.extend(
        (
            "",
            "## Reproducibility",
            "",
            f"- Deterministic decision hash: "
            f"`{summary['deterministic_result_hash']}`",
            f"- Explicit implementation snapshot version edges: "
            f"{len(source_supersessions)}; original registered hashes are "
            "retained and never overwritten.",
            "- The audit manifest records hashes for every frozen input and "
            "machine-readable output.",
            "",
        )
    )
    (OUTPUT_DIR / "audit_report_v3.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _human_attested_review_artifacts() -> dict[str, Any]:
    """Verify exact files covered by the project-owner attestation."""
    attestation_path = ROOT / "HUMAN_REVIEW_ATTESTATION_20260729.json"
    archive_dir = OUTPUT_DIR / "human_attested_automated_drafts_20260729"
    attestation = read_json(attestation_path)
    artifacts: list[dict[str, Any]] = []
    for item in attestation.get("scope", []):
        archived_filename = str(item["archived_filename"])
        path = archive_dir / archived_filename
        expected = str(item["sha256"])
        actual = sha256_file(path) if path.is_file() else ""
        artifacts.append(
            {
                "path": str(path.resolve()),
                "reviewer_role": str(item["reviewer_role"]),
                "decision_type": str(item["decision_type"]),
                "expected_sha256": expected,
                "sha256": actual,
                "hash_matches_attestation": bool(actual == expected),
                "provenance": "human_attested_automated_draft",
            }
        )
    return {
        "attestation_path": str(attestation_path.resolve()),
        "attestation_sha256": sha256_file(attestation_path),
        "artifacts": artifacts,
        "all_hashes_match": bool(artifacts)
        and all(item["hash_matches_attestation"] for item in artifacts),
    }


def _write_prisma_table(prisma: Mapping[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for key, value in prisma.items():
        if key == "exclusions_by_reason":
            continue
        rows.append({"stage": key, "count": value, "reason": ""})
    exclusions = prisma.get("exclusions_by_reason", {})
    if isinstance(exclusions, dict):
        for reason, count in sorted(exclusions.items()):
            rows.append(
                {
                    "stage": "excluded",
                    "count": count,
                    "reason": reason,
                }
            )
    write_csv_iter(
        OUTPUT_DIR / "prisma_flow_v3.csv",
        rows,
        ("stage", "count", "reason"),
    )


def audit(connection: sqlite3.Connection) -> dict[str, Any]:
    """Generate a full evidence audit and enforce all completion gates."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    record_digest_status = _ensure_record_payload_digests(connection)
    counts = current_counts(connection)
    stages = _stage_summary(connection)
    blockers = _completion_blockers(connection, stages)
    attested_review = _human_attested_review_artifacts()
    if not attested_review["all_hashes_match"]:
        blockers.append(
            "One or more human-attested H1/H2 files are missing or no "
            "longer match the project-owner attestation hash."
        )
    deterministic_payload = _deterministic_result_payload(connection)
    result_hash = json_hash(deterministic_payload)
    set_stage(
        connection,
        "audit_complete",
        "complete" if not blockers else "blocked",
        {
            "formal_review_complete": not blockers,
            "blocker_count": len(blockers),
            "deterministic_result_hash": result_hash,
        },
    )
    connection.commit()
    stages = _stage_summary(connection)
    invalidated_directory = OUTPUT_DIR / "invalidated_automated_h1_trial_20260729"
    invalidated_h1_artifacts = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for path in sorted(invalidated_directory.glob("*"))
        if path.is_file()
    ]
    invalidated_qwen_directory = OUTPUT_DIR / "invalidated_local_qwen_review_20260729"
    invalidated_qwen_artifacts = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for path in sorted(invalidated_qwen_directory.glob("*"))
        if path.is_file()
    ]
    unreviewed_h2_directory = OUTPUT_DIR / "unreviewed_automated_h2_drafts_20260729"
    unreviewed_h2_artifacts = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for path in sorted(unreviewed_h2_directory.glob("*"))
        if path.is_file()
    ]
    independent_codex_directory = OUTPUT_DIR / "independent_codex_review_v3"
    independent_codex_artifacts = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
        for path in sorted(independent_codex_directory.glob("*"))
        if path.is_file()
    ]
    independent_review_summary = dict(connection.execute("""
            SELECT COUNT(*) AS run_count,
                   COALESCE(SUM(item_count), 0) AS item_count,
                   COUNT(DISTINCT model || '|' || model_digest)
                       AS distinct_model_builds
            FROM independent_ai_review_runs
            WHERE status = 'complete'
            """).fetchone())
    independent_review_summary["models"] = [dict(row) for row in connection.execute("""
            SELECT model, model_digest, COUNT(*) AS run_count,
                   SUM(item_count) AS item_count
            FROM independent_ai_review_runs
            WHERE status = 'complete'
            GROUP BY model, model_digest
            ORDER BY model, model_digest
            """)]
    final_training_features = _final_training_feature_status(connection)
    retained_dimensions = _rows(
        connection,
        """
        SELECT c.dimension_id, c.label, c.definition,
               d.dimension_role, d.independent_group_count,
               d.selected_feature_ids_json
        FROM candidate_dimensions c
        JOIN dimension_decisions d USING(dimension_id)
        WHERE d.selected = 1
        ORDER BY CASE d.dimension_role
                   WHEN 'predictive' THEN 0
                   WHEN 'opportunity' THEN 1
                   WHEN 'control' THEN 2
                   ELSE 3
                 END,
                 c.dimension_id
        """,
    )
    final_indicators = _rows(
        connection,
        """
        SELECT f.feature_id, f.canonical_name_en, f.label_zh,
               f.formula, f.missing_rule, d.final_role,
               a.valid_count, a.missing_rate, a.unique_count
        FROM indicator_families f
        JOIN feature_decisions d USING(feature_id)
        JOIN feature_data_audit a USING(feature_id)
        WHERE d.final_role != 'excluded'
        ORDER BY CASE d.final_role
                   WHEN 'predictive' THEN 0
                   WHEN 'opportunity' THEN 1
                   WHEN 'control' THEN 2
                   ELSE 3
                 END,
                 f.feature_id
        """,
    )
    source_snapshot_supersessions = _rows(
        connection,
        """
        SELECT old_source_id, new_source_id, old_sha256,
               observed_current_sha256, authorization_source_id, reason
        FROM source_snapshot_supersessions
        ORDER BY old_source_id
        """,
    )
    summary: dict[str, Any] = {
        "schema_version": "3.4.0",
        "generated_at": utc_now(),
        "counts": counts,
        "no_numeric_quota": True,
        "prisma": _prisma(connection),
        "discovery": _discovery_summary(connection),
        "record_payload_digests": record_digest_status,
        "stages": stages,
        "completion_blockers": blockers,
        "formal_review_complete": not blockers,
        "deterministic_result_hash": result_hash,
        "language_scope": "English only",
        "language_geographic_bias_disclosed": True,
        "invalidated_automated_h1_artifacts": invalidated_h1_artifacts,
        "invalidated_local_qwen_review_artifacts": (invalidated_qwen_artifacts),
        "unreviewed_automated_h2_artifacts": unreviewed_h2_artifacts,
        "human_review_attestation": {
            "path": attested_review["attestation_path"],
            "sha256": attested_review["attestation_sha256"],
            "all_hashes_match": attested_review["all_hashes_match"],
        },
        "human_attested_review_artifacts": attested_review["artifacts"],
        "independent_ai_reviews": independent_review_summary,
        "independent_codex_review_artifacts": (independent_codex_artifacts),
        "final_training_features": final_training_features,
        "retained_dimensions": retained_dimensions,
        "final_indicators": final_indicators,
        "source_snapshot_supersessions": source_snapshot_supersessions,
        "reviewer_substitution_amendment": {
            "path": str(
                (ROOT / "protocol_amendment_independent_ai_review_v3.json").resolve()
            ),
            "sha256": sha256_file(
                ROOT / "protocol_amendment_independent_ai_review_v3.json"
            ),
            "separate_codex_task_not_human": True,
            "review_task_id": ("019fabc1-0c6d-7771-92e4-4501e5bee18b"),
        },
        "round12_pragmatic_stop_amendment": {
            "path": str(DISCOVERY_STOP_AMENDMENT_PATH.resolve()),
            "sha256": sha256_file(DISCOVERY_STOP_AMENDMENT_PATH),
            "retrospective_protocol_deviation": True,
            "actual_counts_retained": True,
            "dual_zero_claim_requires_computed_dual_zero": True,
        },
        "round12_external_reporting_clarification": {
            "path": str(ROUND12_EXTERNAL_REPORTING_CLARIFICATION_PATH.resolve()),
            "sha256": sha256_file(ROUND12_EXTERNAL_REPORTING_CLARIFICATION_PATH),
            "external_short_label": "post-freeze expansion 0/0",
            "post_freeze_new_term_families": 0,
            "post_freeze_new_indicator_families": 0,
            "round12_actual_counts_retained": True,
        },
    }
    _export_evidence_tables(connection)
    _write_prisma_table(summary["prisma"])
    write_json(OUTPUT_DIR / "audit_summary_v3.json", summary)
    _write_report(summary)
    output_paths = sorted(
        path
        for path in OUTPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix != ".sqlite3"
        and path.name
        not in {
            "evidence_derived_v3.sqlite3",
            "evidence_derived_v3.sqlite3-wal",
            "evidence_derived_v3.sqlite3-shm",
            "audit_manifest_v3.json",
        }
    )
    manifest = {
        "schema_version": "3.4.0",
        "decision_hash": result_hash,
        "sources": _rows(
            connection,
            """
            SELECT source_id, path, sha256, role
            FROM source_snapshots ORDER BY source_id
            """,
        ),
        "outputs": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in output_paths
        ],
        "invalidated_artifacts": [
            *invalidated_h1_artifacts,
            *invalidated_qwen_artifacts,
        ],
        "unreviewed_artifacts": unreviewed_h2_artifacts,
        "human_review_attestation": summary["human_review_attestation"],
        "human_attested_review_artifacts": attested_review["artifacts"],
        "independent_ai_reviews": independent_review_summary,
        "independent_codex_review_artifacts": (independent_codex_artifacts),
        "final_training_features": final_training_features,
        "reviewer_substitution_amendment": summary["reviewer_substitution_amendment"],
    }
    manifest["manifest_hash"] = json_hash(manifest)
    write_json(OUTPUT_DIR / "audit_manifest_v3.json", manifest)
    return summary
