from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from common import (
    ROOT,
    cohen_kappa,
    gwet_ac1,
    iter_csv,
    json_hash,
    normalize_doi,
    normalize_term,
    or_block,
    parse_bool,
    raw_agreement,
    read_csv,
    read_json,
    sha256_bytes,
    sha256_file,
    term_match_key,
    utc_now,
    write_csv,
    write_csv_iter,
    write_json,
)
from database import (
    HUMAN_ATTESTED_AUTOMATED_DRAFT_HASHES,
    invalidate_stages,
    log_event,
    require_complete,
    set_stage,
    snapshot_import_file,
)
from providers import (
    batch_openalex_seed_matches,
    inventory_physical_queries,
    openalex_api_keys,
    query_definition_hash,
    retrieve_physical_query,
)


PROTOCOL_PATH = ROOT / "protocol_v3.json"
BOOTSTRAP_PATH = ROOT / "bootstrap_query_v3.json"
SATURATION_PROTOCOL_PATH = ROOT / "saturation_protocol_v3.json"
RULES_PATH = ROOT / "screening_rules_v3.json"
V2_ROOT = ROOT.parent / "expanded_review_v2"
V2_EVIDENCE_PATH = V2_ROOT / "outputs" / "literature_evidence_v2.json"
V2_DATABASE_PATH = V2_ROOT / "outputs" / "expanded_search.sqlite3"
V2_INDICATOR_PATH = V2_ROOT / "outputs" / "indicator_catalog_v2.csv"
FORMAL_SEARCH_FRAME_PATH = ROOT / "outputs" / "frozen_search_frame_v3.json"
HUMAN_REVIEW_ATTESTATION_PATH = (
    ROOT / "HUMAN_REVIEW_ATTESTATION_20260729.json"
)
SEARCH_FRAME_DOWNSTREAM_STAGES = (
    "terms_coded",
    "search_frame_derived",
    "search_frame_validated",
    "search_frame_frozen",
    "formal_retrieval_complete",
    "literature_screened",
    "indicators_extracted",
    "dimensions_derived",
    "features_selected",
    "audit_complete",
)

TERM_FIELDS = (
    "term_id",
    "source_record_key",
    "source_id",
    "source_type",
    "source_language_status",
    "source_language_evidence",
    "verbatim_term",
    "location",
    "evidence_span",
    "proposed_role",
    "status",
    "exclusion_reason",
)
TERM_CODING_FIELDS = (
    "term_id",
    "verbatim_term",
    "source_type",
    "coder_role",
    "canonical_term",
    "term_family_label",
    "term_relation",
    "search_domain_label",
    "search_domain_definition",
    "query_family_label",
    "cross_domain",
    "decision",
    "reason",
)
TERM_RELATIONS = {
    "canonical",
    "synonym",
    "abbreviation",
    "full_form",
    "historical_name",
    "morphological_variant",
    "parameter_variant",
}
NONAUTHORIZING_TERM_SOURCE_TYPES = {
    "pilot_v2_indicator",
    "pilot_v2_literature",
    "development_seed_hint",
}
PRESS_FIELDS = (
    "logical_query_id",
    "search_domain_id",
    "family_label",
    "logical_expression",
    "reviewer_role",
    "concepts_complete",
    "boolean_logic_valid",
    "spelling_valid",
    "phrases_valid",
    "limits_justified",
    "covered_by_logical_query_id",
    "logical_coverage_verified",
    "result_set_coverage_verified",
    "independent_construct_role",
    "decision",
    "notes",
)
SEED_FIELDS = (
    "seed_id",
    "doi",
    "citation",
    "publication_year",
    "language",
    "seed_role",
    "supplied_by",
    "hidden_during_development",
    "eligibility_status",
    "nonrecall_reason",
)
HIDDEN_SEED_SEARCH_LOG_FIELDS = (
    "search_run_id",
    "reviewer_role",
    "route",
    "source_name",
    "exact_query_or_seed",
    "executed_at",
    "retrieved_count",
    "screened_count",
    "eligible_seed_count",
    "eligible_seed_dois",
    "completion_status",
    "notes",
)
HIDDEN_SEED_SEARCH_ROUTES = {
    "independent_review_search",
    "backward_citation_tracking",
    "forward_citation_tracking",
}
SEED_SUPPLEMENT_FIELDS = (
    "seed_id",
    "doi",
    "title",
    "abstract",
    "publication_year",
    "work_type",
    "language",
    "source_url",
    "supplied_by",
)


def _assert_not_frozen(connection: sqlite3.Connection) -> None:
    frozen = connection.execute(
        "SELECT value FROM metadata WHERE key = 'search_frame_frozen_hash'"
    ).fetchone()
    if frozen is not None:
        raise RuntimeError(
            "The search frame is frozen. Use reopen-search-frame with an "
            "H2-authorized reason before changing terms, domains, or queries."
        )


def _register_snapshot(
    connection: sqlite3.Connection,
    source_id: str,
    path: Path,
    role: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    existing = connection.execute(
        """
        SELECT path, sha256, role FROM source_snapshots
        WHERE source_id = ?
        """,
        (source_id,),
    ).fetchone()
    frozen = connection.execute(
        """
        SELECT 1 FROM metadata
        WHERE key = 'search_frame_frozen_hash'
        """
    ).fetchone()
    if (
        existing is not None
        and frozen is not None
        and (
            str(existing["sha256"]) != digest
            or str(existing["role"]) != role
        )
    ):
        raise RuntimeError(
            "A registered source snapshot changed after search-frame freeze: "
            f"{source_id}. Reopen the frame with H2 authorization or restore "
            "the frozen file; init cannot rewrite its audit baseline."
        )
    connection.execute(
        """
        INSERT INTO source_snapshots(
            source_id, path, sha256, role, imported_at
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(source_id) DO UPDATE SET
            path = excluded.path,
            sha256 = excluded.sha256,
            role = excluded.role,
            imported_at = excluded.imported_at
        """,
        (source_id, str(path.resolve()), digest, role, utc_now()),
    )


def _bootstrap_expression(config: Mapping[str, Any]) -> str:
    return " AND ".join(
        (
            or_block(config["object_terms"]),
            or_block(config["target_terms"]),
            or_block(config["evidence_terms"]),
        )
    )


def _provider_filter(config: Mapping[str, Any]) -> str:
    filters = config["filters"]
    types = "|".join(str(value) for value in filters["types"])
    values = [
        f"to_publication_date:{filters['to_publication_date']}",
        f"type:{types}",
    ]
    from_date = filters.get("from_publication_date")
    if from_date:
        values.insert(0, f"from_publication_date:{from_date}")
    return ",".join(values)


def initialize_project(connection: sqlite3.Connection) -> Dict[str, Any]:
    """Register frozen inputs, development seeds, and bootstrap query."""
    protocol = read_json(PROTOCOL_PATH)
    bootstrap = read_json(BOOTSTRAP_PATH)
    attestation = read_json(HUMAN_REVIEW_ATTESTATION_PATH)
    attested_hashes = {
        str(item.get("sha256") or "")
        for item in attestation.get("scope", [])
        if isinstance(item, dict) and str(item.get("sha256") or "")
    }
    if attested_hashes != set(HUMAN_ATTESTED_AUTOMATED_DRAFT_HASHES):
        raise RuntimeError(
            "Human-review attestation hashes do not match the import "
            "provenance allowlist"
        )
    for source_id, path, role in (
        ("protocol_v3", PROTOCOL_PATH, "frozen_protocol"),
        (
            "execution_goal_v3",
            ROOT / "EXECUTION_GOAL_V3.md",
            "completion_contract",
        ),
        (
            "methods_evidence_saturation_v3",
            ROOT / "METHODS_EVIDENCE_SATURATION_V3.md",
            "reporting_template",
        ),
        (
            "human_review_attestation_20260729",
            HUMAN_REVIEW_ATTESTATION_PATH,
            "human_review_attestation",
        ),
        ("bootstrap_query_v3", BOOTSTRAP_PATH, "bootstrap_definition"),
        (
            "saturation_protocol_v3",
            SATURATION_PROTOCOL_PATH,
            "frozen_saturation_protocol",
        ),
        ("screening_rules_v3", RULES_PATH, "frozen_selection_rules"),
        ("v2_evidence_53", V2_EVIDENCE_PATH, "development_seed_source"),
        ("v2_pilot_database", V2_DATABASE_PATH, "pilot_term_source"),
        ("v2_indicator_catalog", V2_INDICATOR_PATH, "pilot_term_source"),
        ("code_common", ROOT / "common.py", "implementation"),
        ("code_database", ROOT / "database.py", "implementation"),
        ("code_providers", ROOT / "providers.py", "implementation"),
        ("code_coding", ROOT / "coding.py", "implementation"),
        ("code_screening", ROOT / "screening.py", "implementation"),
        ("code_indicators", ROOT / "indicators.py", "implementation"),
        ("code_retrieval", ROOT / "retrieval.py", "implementation"),
        ("code_saturation", ROOT / "saturation.py", "implementation"),
        ("code_local_ai", ROOT / "local_ai.py", "implementation"),
        ("code_handoff", ROOT / "handoff.py", "implementation"),
        (
            "code_draft_h2_assistance",
            ROOT / "draft_h2_assistance.py",
            "review_assistance_implementation",
        ),
        (
            "code_human_review_cli",
            ROOT / "human_review_cli.py",
            "implementation",
        ),
        ("code_reporting", ROOT / "reporting.py", "implementation"),
        ("code_pipeline", ROOT / "pipeline.py", "implementation"),
        ("code_tests_v3", ROOT / "tests_v3.py", "verification"),
    ):
        _register_snapshot(connection, source_id, path, role)
    saturation_protocol = read_json(SATURATION_PROTOCOL_PATH)
    snapshot = saturation_protocol["snapshot"]
    snapshot_root = Path(str(snapshot["root"]))
    snapshot_manifest = Path(str(snapshot["works_manifest"]))
    if not snapshot_root.exists():
        raise FileNotFoundError(snapshot_root)
    if not snapshot_manifest.exists():
        raise FileNotFoundError(snapshot_manifest)
    snapshot_manifest_payload = read_json(snapshot_manifest)
    snapshot_entries = snapshot_manifest_payload.get("entries", [])
    snapshot_meta = snapshot_manifest_payload.get("meta", {})
    if not isinstance(snapshot_entries, list) or not isinstance(
        snapshot_meta, dict
    ):
        raise ValueError("OpenAlex works manifest has an invalid structure")
    snapshot_record_count = int(
        snapshot_meta.get("record_count")
        or sum(
            int(entry.get("meta", {}).get("record_count") or 0)
            for entry in snapshot_entries
            if isinstance(entry, dict)
        )
    )
    snapshot_content_length = int(
        snapshot_meta.get("content_length")
        or sum(
            int(entry.get("meta", {}).get("content_length") or 0)
            for entry in snapshot_entries
            if isinstance(entry, dict)
        )
    )
    snapshot_updated_dates = [
        str(entry.get("url") or "").split("updated_date=", maxsplit=1)[1]
        .split("/", maxsplit=1)[0]
        for entry in snapshot_entries
        if isinstance(entry, dict)
        and "updated_date=" in str(entry.get("url") or "")
    ]
    observed_snapshot_max_date = max(snapshot_updated_dates, default="")
    configured_snapshot_max_date = str(snapshot["observed_max_updated_date"])
    if observed_snapshot_max_date != configured_snapshot_max_date:
        raise ValueError(
            "OpenAlex snapshot maximum date changed: "
            f"manifest={observed_snapshot_max_date}, "
            f"protocol={configured_snapshot_max_date}"
        )
    connection.execute(
        """
        INSERT INTO local_snapshot_sources(
            snapshot_id, root_path, manifest_path, manifest_sha256,
            part_count, record_count, content_length_bytes,
            maximum_updated_date, role, registered_at
        ) VALUES ('openalex_local_works', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(snapshot_id) DO UPDATE SET
            root_path = excluded.root_path,
            manifest_path = excluded.manifest_path,
            manifest_sha256 = excluded.manifest_sha256,
            part_count = excluded.part_count,
            record_count = excluded.record_count,
            content_length_bytes = excluded.content_length_bytes,
            maximum_updated_date = excluded.maximum_updated_date,
            role = excluded.role,
            registered_at = excluded.registered_at
        """,
        (
            str(snapshot_root.resolve()),
            str(snapshot_manifest.resolve()),
            sha256_file(snapshot_manifest),
            len(snapshot_entries),
            snapshot_record_count,
            snapshot_content_length,
            observed_snapshot_max_date,
            str(snapshot["role"]),
            utc_now(),
        ),
    )
    evidence = read_json(V2_EVIDENCE_PATH)
    records = evidence.get("records")
    if not isinstance(records, list):
        raise ValueError("v2 evidence file has no records array")
    seed_count = 0
    for item in records:
        if not isinstance(item, dict):
            continue
        doi = normalize_doi(item.get("doi"))
        if not doi:
            continue
        seed_id = str(item.get("source_id") or "").strip()
        if not seed_id:
            seed_id = "DEV_" + sha256_bytes(doi.encode("utf-8"))[:12]
        connection.execute(
            """
            INSERT INTO evidence_seeds(
                seed_id, doi, citation, publication_year, language,
                seed_role, supplied_by, hidden_during_development,
                eligibility_status
            ) VALUES (?, ?, ?, ?, 'en', 'development', 'v2_evidence',
                      0, 'eligible')
            ON CONFLICT(seed_id) DO UPDATE SET
                doi = excluded.doi,
                citation = excluded.citation,
                publication_year = excluded.publication_year
            """,
            (
                seed_id,
                doi,
                str(item.get("citation") or ""),
                item.get("year"),
            ),
        )
        seed_count += 1
    expression = _bootstrap_expression(bootstrap)
    filter_expression = _provider_filter(bootstrap)
    logical_id = str(bootstrap["logical_query_id"])
    query_hash = query_definition_hash(expression, filter_expression)
    connection.execute(
        """
        INSERT INTO logical_queries(
            logical_query_id, query_version, search_domain_id,
            family_label, logical_expression, object_terms_json,
            domain_terms_json, context_terms_json, status,
            archive_reason, press_status, query_hash
        ) VALUES (?, 1, 'BOOTSTRAP', 'domain_agnostic', ?, ?, ?, ?,
                  'bootstrap', '', 'not_applicable', ?)
        ON CONFLICT(logical_query_id) DO UPDATE SET
            logical_expression = excluded.logical_expression,
            object_terms_json = excluded.object_terms_json,
            domain_terms_json = excluded.domain_terms_json,
            context_terms_json = excluded.context_terms_json,
            query_hash = excluded.query_hash
        """,
        (
            logical_id,
            expression,
            json.dumps(bootstrap["object_terms"], ensure_ascii=False),
            json.dumps(bootstrap["target_terms"], ensure_ascii=False),
            json.dumps(bootstrap["evidence_terms"], ensure_ascii=False),
            query_hash,
        ),
    )
    physical_id = f"{logical_id}__P001"
    connection.execute(
        """
        INSERT INTO physical_queries(
            physical_query_id, logical_query_id, provider, expression,
            filter_expression, status, query_hash
        ) VALUES (?, ?, 'OpenAlex', ?, ?, 'active', ?)
        ON CONFLICT(physical_query_id) DO UPDATE SET
            expression = excluded.expression,
            filter_expression = excluded.filter_expression,
            query_hash = excluded.query_hash
        """,
        (
            physical_id,
            logical_id,
            expression,
            filter_expression,
            query_hash,
        ),
    )
    pilot_count = seed_pilot_terms(connection)
    development_term_count = seed_development_evidence_terms(connection)
    connection.execute(
        """
        INSERT INTO metadata(key, value) VALUES ('protocol_id', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (str(protocol["protocol_id"]),),
    )
    set_stage(
        connection,
        "initialized",
        "complete",
        {
            "development_seeds": seed_count,
            "pilot_terms": pilot_count,
            "development_evidence_terms": development_term_count,
            "bootstrap_logical_queries": 1,
            "bootstrap_physical_queries": 1,
        },
    )
    connection.commit()
    return {
        "development_seeds": seed_count,
        "pilot_terms": pilot_count,
        "development_evidence_terms": development_term_count,
        "bootstrap_query_id": physical_id,
    }


def seed_pilot_terms(connection: sqlite3.Connection) -> int:
    """Register v2 indicator names as supplementary, non-authorizing terms."""
    rows = read_csv(V2_INDICATOR_PATH)
    count = 0
    for row in rows:
        verbatim = str(row.get("name") or "").strip()
        if not verbatim:
            continue
        source_id = str(row.get("feature_id") or "")
        identity = f"pilot|{source_id}|{verbatim}"
        term_id = "PILOT_" + sha256_bytes(
            identity.encode("utf-8")
        )[:16].upper()
        connection.execute(
            """
            INSERT OR IGNORE INTO raw_terms(
                term_id, source_record_key, source_id, source_type,
                source_language_status, source_language_evidence,
                verbatim_term, normalized_term, match_key, location,
                evidence_span, proposed_role, status, exclusion_reason
            ) VALUES (?, '', ?, 'pilot_v2_indicator', 'en',
                      'English catalog term; source language still requires '
                      || 'verification for formal evidence',
                      ?, ?, ?,
                      'v2 indicator catalog name', ?, ?, 'active', '')
            """,
            (
                term_id,
                source_id,
                verbatim,
                normalize_term(verbatim.replace("_", " ")),
                term_match_key(verbatim.replace("_", " ")),
                str(row.get("evidence_summary") or ""),
                str(row.get("scope_role") or ""),
            ),
        )
        count += 1
    return count


def seed_development_evidence_terms(
    connection: sqlite3.Connection,
) -> int:
    """Register v2-derived development hints without authorizing K/Q."""
    evidence = read_json(V2_EVIDENCE_PATH)
    count = 0
    for source in evidence.get("records", []):
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("source_id") or "").strip()
        doi = normalize_doi(source.get("doi"))
        for phrase_value in source.get("formula_authorization", []):
            phrase = str(phrase_value or "").strip()
            if not phrase:
                continue
            identity = f"development|{source_id}|{doi}|{phrase}"
            term_id = "DEVTERM_" + sha256_bytes(
                identity.encode("utf-8")
            )[:16].upper()
            connection.execute(
                """
                INSERT OR IGNORE INTO raw_terms(
                    term_id, source_record_key, source_id, source_type,
                    source_language_status, source_language_evidence,
                    verbatim_term, normalized_term, match_key, location,
                    evidence_span, proposed_role, status, exclusion_reason
                ) VALUES (?, ?, ?, 'development_seed_hint', 'en',
                          'Derived v2 development hint; direct source '
                          || 'verification is still required for K/Q',
                          ?, ?, ?, 'formula_authorization field', ?,
                          'indicator_or_measure', 'active', '')
                """,
                (
                    term_id,
                    f"seed:{source_id}",
                    doi or source_id,
                    phrase,
                    normalize_term(phrase),
                    term_match_key(phrase),
                    phrase,
                ),
            )
            count += 1
    connection.execute(
        """
        UPDATE raw_terms
        SET source_type = 'development_seed_hint',
            source_language_evidence =
                'Derived v2 development hint; direct source verification '
                || 'is still required for K/Q'
        WHERE term_id LIKE 'DEVTERM_%'
          AND source_type = 'development_seed'
        """
    )
    return count


def bootstrap_query_ids(connection: sqlite3.Connection) -> List[str]:
    """Return only domain-agnostic bootstrap physical queries."""
    return [
        str(row[0])
        for row in connection.execute(
            """
            SELECT physical_query_id
            FROM physical_queries
            WHERE logical_query_id = 'B0001_DOMAIN_AGNOSTIC_BOOTSTRAP'
            ORDER BY physical_query_id
            """
        )
    ]


def mark_bootstrap_inventory(
    connection: sqlite3.Connection,
) -> Dict[str, int]:
    """Run and record the bootstrap inventory."""
    require_complete(connection, ["initialized"])
    totals = inventory_physical_queries(
        connection,
        bootstrap_query_ids(connection),
        "bootstrap_inventory",
    )
    set_stage(
        connection,
        "bootstrap_inventory_complete",
        "complete",
        {"reported_total": sum(totals.values()), "queries": totals},
    )
    connection.commit()
    return totals


def mark_bootstrap_retrieval_stage(
    connection: sqlite3.Connection,
) -> Dict[str, Any]:
    """Set bootstrap completion from persisted cursor checkpoints."""
    query_ids = bootstrap_query_ids(connection)
    rows = connection.execute(
        f"""
        SELECT physical_query_id, complete, reported_total, retrieved_rows,
               unique_hits, pages, error
        FROM query_runs
        WHERE provider = 'OpenAlex' AND run_role = 'bootstrap'
          AND physical_query_id IN ({','.join('?' for _ in query_ids)})
        ORDER BY physical_query_id
        """,
        query_ids,
    ).fetchall()
    complete = len(rows) == len(query_ids) and all(row["complete"] for row in rows)
    details = {
        "physical_queries": len(query_ids),
        "complete_queries": sum(int(row["complete"]) for row in rows),
        "retrieved_rows": sum(int(row["retrieved_rows"]) for row in rows),
        "unique_records": connection.execute(
            """
            SELECT COUNT(DISTINCT record_key)
            FROM query_hits
            WHERE provider = 'OpenAlex' AND run_role = 'bootstrap'
            """
        ).fetchone()[0],
    }
    set_stage(
        connection,
        "bootstrap_retrieval_complete",
        "complete" if complete else "ready",
        details,
    )
    connection.commit()
    return details


def export_term_extraction(
    connection: sqlite3.Connection,
    output_path: Path,
) -> int:
    """Export a source-preserving term extraction worksheet."""
    require_complete(connection, ["bootstrap_retrieval_complete"])
    fields = list(TERM_FIELDS) + ["source_title", "source_abstract"]
    saturation_mode = (
        connection.execute(
            "SELECT COUNT(*) FROM discovery_queries WHERE status = 'active'"
        ).fetchone()[0]
        > 0
    )

    def iter_rows() -> Iterable[Dict[str, Any]]:
        discovery_clause = (
            """
            AND EXISTS (
                SELECT 1 FROM discovery_hits d
                WHERE d.record_key = r.record_key
                  AND d.review_round > 0
                  AND d.review_status = 'include'
            )
            """
            if saturation_mode
            else ""
        )
        records = connection.execute(
            f"""
            SELECT DISTINCT r.record_key, r.doi, r.title, r.abstract,
                            r.language
            FROM records r
            JOIN query_hits q
              ON q.provider = r.provider AND q.record_key = r.record_key
            WHERE q.run_role = 'bootstrap'
            {discovery_clause}
            ORDER BY r.record_key
            """
        )
        for record in records:
            yield {
                "term_id": "",
                "source_record_key": record["record_key"],
                "source_id": record["doi"],
                "source_type": "bootstrap_literature",
                "source_language_status": record["language"],
                "source_language_evidence": (
                    f"OpenAlex language={record['language']}; "
                    "verify against title/abstract"
                ),
                "verbatim_term": "",
                "location": "title|abstract|full_text",
                "evidence_span": "",
                "proposed_role": "",
                "status": "active",
                "exclusion_reason": "",
                "source_title": record["title"],
                "source_abstract": record["abstract"],
            }
        for seed in connection.execute(
            """
            SELECT e.seed_id, e.doi, e.citation,
                   COALESCE(
                       (
                           SELECT r.title FROM records r
                           WHERE r.doi = e.doi
                           ORDER BY r.provider = 'OpenAlex' DESC
                           LIMIT 1
                       ),
                       e.citation
                   ) AS source_title,
                   COALESCE(
                       (
                           SELECT r.abstract FROM records r
                           WHERE r.doi = e.doi
                           ORDER BY r.provider = 'OpenAlex' DESC
                           LIMIT 1
                       ),
                       ''
                   ) AS source_abstract
            FROM evidence_seeds e
            WHERE seed_role = 'development'
            ORDER BY seed_id
            """
        ):
            yield {
                "term_id": "",
                "source_record_key": f"seed:{seed['seed_id']}",
                "source_id": seed["doi"],
                "source_type": "development_seed",
                "source_language_status": "en",
                "source_language_evidence": (
                    "Frozen development evidence is marked English"
                ),
                "verbatim_term": "",
                "location": "title|abstract|full_text",
                "evidence_span": "",
                "proposed_role": "",
                "status": "active",
                "exclusion_reason": "",
                "source_title": seed["source_title"],
                "source_abstract": seed["source_abstract"],
            }
        if not saturation_mode:
            pilot = sqlite3.connect(
                f"file:{V2_DATABASE_PATH.resolve()}?mode=ro",
                uri=True,
            )
            pilot.row_factory = sqlite3.Row
            try:
                for record in pilot.execute(
                    """
                    SELECT record_key, doi, title, abstract
                    FROM works
                    ORDER BY record_key
                    """
                ):
                    yield {
                        "term_id": "",
                        "source_record_key": f"v2:{record['record_key']}",
                        "source_id": record["doi"],
                        "source_type": "pilot_v2_literature",
                        "source_language_status": "unknown",
                        "source_language_evidence": "",
                        "verbatim_term": "",
                        "location": "title|abstract",
                        "evidence_span": "",
                        "proposed_role": "",
                        "status": "active",
                        "exclusion_reason": "",
                        "source_title": record["title"],
                        "source_abstract": record["abstract"],
                    }
            finally:
                pilot.close()

    return write_csv_iter(output_path, iter_rows(), fields)


def _term_identity(row: Mapping[str, Any]) -> str:
    payload = "|".join(
        (
            str(row.get("source_record_key") or ""),
            str(row.get("source_id") or ""),
            str(row.get("source_type") or ""),
            str(row.get("verbatim_term") or ""),
            str(row.get("location") or ""),
        )
    )
    return "TERM_" + sha256_bytes(payload.encode("utf-8"))[:16].upper()


def import_terms(
    connection: sqlite3.Connection,
    input_path: Path,
) -> Dict[str, int]:
    """Import extracted English terms and their exact source spans."""
    _assert_not_frozen(connection)
    snapshot_path = snapshot_import_file(
        connection,
        input_path,
        "raw_terms",
    )
    rows = iter_csv(snapshot_path)
    counts = {"active": 0, "excluded": 0}
    allowed_sources = {
        "bootstrap_literature",
        "development_seed",
        "development_seed_hint",
        "citation_tracking",
        "pilot_v2_indicator",
        "pilot_v2_literature",
        "manual_openalex_supplement",
    }
    for row in rows:
        verbatim = str(row.get("verbatim_term") or "").strip()
        if not verbatim:
            continue
        source_type = str(row.get("source_type") or "").strip()
        if source_type not in allowed_sources:
            raise ValueError(f"Invalid source_type: {source_type}")
        status = str(row.get("status") or "active").strip().casefold()
        if status not in {"active", "excluded"}:
            raise ValueError(f"Invalid term status: {status}")
        exclusion_reason = str(row.get("exclusion_reason") or "").strip()
        if status == "excluded" and not exclusion_reason:
            raise ValueError("Excluded terms require an exclusion reason")
        evidence_span = str(row.get("evidence_span") or "").strip()
        language_status = str(
            row.get("source_language_status") or ""
        ).strip().casefold()
        language_evidence = str(
            row.get("source_language_evidence") or ""
        ).strip()
        if language_status not in {"en", "non_en", "unknown"}:
            raise ValueError(
                f"Invalid source_language_status: {language_status}"
            )
        if status == "active" and (
            language_status != "en" or not language_evidence
        ):
            raise ValueError(
                f"Active term {verbatim!r} requires verified English source"
            )
        if language_status == "non_en" and (
            status != "excluded"
            or exclusion_reason != "E_LANGUAGE_NON_ENGLISH"
        ):
            raise ValueError(
                "Non-English source terms must be excluded with "
                "E_LANGUAGE_NON_ENGLISH"
            )
        if status == "active" and not evidence_span:
            raise ValueError(
                f"Active term {verbatim!r} requires a source evidence span"
            )
        source_record_key = str(
            row.get("source_record_key") or ""
        ).strip()
        if status == "active" and source_type in {
            "bootstrap_literature",
            "development_seed",
            "citation_tracking",
            "manual_openalex_supplement",
        }:
            if source_type == "development_seed":
                source_doi = normalize_doi(row.get("source_id"))
                source_record = connection.execute(
                    """
                    SELECT title, abstract, language FROM records
                    WHERE doi = ?
                    ORDER BY provider = 'OpenAlex' DESC
                    LIMIT 1
                    """,
                    (source_doi,),
                ).fetchone()
            else:
                source_record = connection.execute(
                    """
                    SELECT title, abstract, language FROM records
                    WHERE record_key = ?
                    """,
                    (source_record_key,),
                ).fetchone()
            if source_record is None:
                raise ValueError(
                    f"Source-linked term has no hydrated v3 record: "
                    f"{source_record_key}/{verbatim}"
                )
            if str(source_record["language"]).casefold() != "en":
                raise ValueError(
                    f"Active term source is not English: {source_record_key}"
                )
            location = str(row.get("location") or "").strip().casefold()
            if location not in {"title", "abstract"}:
                raise ValueError(
                    f"Literature term location must be title or abstract: "
                    f"{source_record_key}/{verbatim}"
                )
            source_text = str(source_record[location] or "")
            if evidence_span.casefold() not in source_text.casefold():
                raise ValueError(
                    f"Term evidence is not an exact {location} span: "
                    f"{source_record_key}/{verbatim}"
                )
            if verbatim.casefold() not in source_text.casefold():
                raise ValueError(
                    f"Verbatim term is absent from {location}: "
                    f"{source_record_key}/{verbatim}"
                )
        term_id = str(row.get("term_id") or "").strip() or _term_identity(row)
        normalized = normalize_term(verbatim)
        connection.execute(
            """
            INSERT INTO raw_terms(
                term_id, source_record_key, source_id, source_type,
                source_language_status, source_language_evidence,
                verbatim_term, normalized_term, match_key, location,
                evidence_span, proposed_role, status, exclusion_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(term_id) DO UPDATE SET
                source_record_key = excluded.source_record_key,
                source_id = excluded.source_id,
                source_type = excluded.source_type,
                source_language_status = excluded.source_language_status,
                source_language_evidence =
                    excluded.source_language_evidence,
                verbatim_term = excluded.verbatim_term,
                normalized_term = excluded.normalized_term,
                match_key = excluded.match_key,
                location = excluded.location,
                evidence_span = excluded.evidence_span,
                proposed_role = excluded.proposed_role,
                status = excluded.status,
                exclusion_reason = excluded.exclusion_reason
            """,
            (
                term_id,
                source_record_key,
                str(row.get("source_id") or "").strip(),
                source_type,
                language_status,
                language_evidence,
                verbatim,
                normalized,
                term_match_key(verbatim),
                str(row.get("location") or "").strip(),
                evidence_span,
                str(row.get("proposed_role") or "").strip(),
                status,
                exclusion_reason,
            ),
        )
        counts[status] += 1
    log_event(
        connection,
        "term_import",
        "file",
        str(snapshot_path.resolve()),
        counts,
    )
    if sum(counts.values()):
        invalidate_stages(
            connection,
            SEARCH_FRAME_DOWNSTREAM_STAGES,
            "raw term evidence changed",
        )
    connection.commit()
    return counts


def export_term_coding(
    connection: sqlite3.Connection,
    output_path: Path,
    reviewer_role: str | None = None,
) -> int:
    """Export independent AI/H1 or adjudicating H2 term-coding rows."""
    require_complete(connection, ["bootstrap_retrieval_complete"])
    if reviewer_role is None:
        raise ValueError(
            "Export one reviewer at a time to preserve independent coding"
        )
    roles = [reviewer_role]
    if any(role not in {"AI", "H1", "H2"} for role in roles):
        raise ValueError("reviewer_role must be AI, H1, or H2")
    evidence_fields = (
        "source_id",
        "source_location",
        "source_evidence_span",
        "proposed_role",
    )
    h2_comparison_fields = (
        "ai_decision",
        "ai_canonical_term",
        "ai_term_family_label",
        "ai_term_relation",
        "ai_search_domain_label",
        "ai_search_domain_definition",
        "ai_query_family_label",
        "ai_cross_domain",
        "ai_reason",
        "h1_decision",
        "h1_canonical_term",
        "h1_term_family_label",
        "h1_term_relation",
        "h1_search_domain_label",
        "h1_search_domain_definition",
        "h1_query_family_label",
        "h1_cross_domain",
        "h1_reason",
    )
    output_fields = list(TERM_CODING_FIELDS) + list(evidence_fields)
    if reviewer_role == "H2":
        output_fields.extend(h2_comparison_fields)
    rows: List[Dict[str, Any]] = []
    terms = connection.execute(
        """
        SELECT term_id, verbatim_term, source_type, source_id, location,
               evidence_span, proposed_role
        FROM raw_terms
        WHERE status = 'active'
        ORDER BY term_id
        """
    ).fetchall()
    for term in terms:
        for role in roles:
            codes: Dict[str, sqlite3.Row] = {}
            if role == "H2":
                codes = {
                    str(row["coder_role"]): row
                    for row in connection.execute(
                        """
                        SELECT * FROM term_coding
                        WHERE term_id = ?
                          AND coder_role IN ('AI', 'H1')
                        """,
                        (term["term_id"],),
                    )
                }
                if "AI" not in codes or "H1" not in codes:
                    raise RuntimeError(
                        "H2 term adjudication export requires complete "
                        f"AI and H1 coding; missing={term['term_id']}"
                    )
                if (
                    _coding_signature(codes["AI"])
                    == _coding_signature(codes["H1"])
                    and codes["AI"]["decision"] == "exclude"
                ):
                    continue
            output: Dict[str, Any] = {
                "term_id": term["term_id"],
                "verbatim_term": term["verbatim_term"],
                "source_type": term["source_type"],
                "coder_role": role,
                "canonical_term": "",
                "term_family_label": "",
                "term_relation": "",
                "search_domain_label": "",
                "search_domain_definition": "",
                "query_family_label": "",
                "cross_domain": "false",
                "decision": "",
                "reason": "",
                "source_id": term["source_id"],
                "source_location": term["location"],
                "source_evidence_span": term["evidence_span"],
                "proposed_role": term["proposed_role"],
            }
            if role == "H2":
                for source_role in ("AI", "H1"):
                    code = codes[source_role]
                    prefix = source_role.casefold()
                    output.update(
                        {
                            f"{prefix}_decision": code["decision"],
                            f"{prefix}_canonical_term": code[
                                "canonical_term"
                            ],
                            f"{prefix}_term_family_label": code[
                                "term_family_label"
                            ],
                            f"{prefix}_term_relation": code[
                                "term_relation"
                            ],
                            f"{prefix}_search_domain_label": code[
                                "search_domain_label"
                            ],
                            f"{prefix}_search_domain_definition": code[
                                "search_domain_definition"
                            ],
                            f"{prefix}_query_family_label": code[
                                "query_family_label"
                            ],
                            f"{prefix}_cross_domain": bool(
                                code["cross_domain"]
                            ),
                            f"{prefix}_reason": code["reason"],
                        }
                    )
            rows.append(output)
    write_csv(output_path, rows, output_fields)
    return len(rows)


def import_term_coding(
    connection: sqlite3.Connection,
    input_path: Path,
) -> int:
    """Import term classifications without inferring missing human codes."""
    _assert_not_frozen(connection)
    snapshot_path = snapshot_import_file(
        connection,
        input_path,
        "term_coding",
    )
    rows = iter_csv(snapshot_path)
    imported = 0
    submission_role = ""
    for row in rows:
        term_id = str(row.get("term_id") or "").strip()
        role = str(row.get("coder_role") or "").strip().upper()
        decision = str(row.get("decision") or "").strip().casefold()
        if not term_id or not decision:
            continue
        if role not in {"AI", "H1", "H2"}:
            raise ValueError(f"Invalid coder role: {role}")
        if submission_role and role != submission_role:
            raise ValueError(
                "One term-coding import cannot mix AI, H1, and H2 roles"
            )
        submission_role = role
        if role == "H1" and any(
            str(field).casefold().startswith(("ai_", "h2_"))
            for field in row
        ):
            raise ValueError(
                "Blind H1 term import refuses AI/H2 comparison columns"
            )
        if decision not in {"include", "exclude"}:
            raise ValueError(f"Invalid term decision: {decision}")
        exists = connection.execute(
            "SELECT 1 FROM raw_terms WHERE term_id = ? AND status = 'active'",
            (term_id,),
        ).fetchone()
        if exists is None:
            raise ValueError(f"Unknown active term: {term_id}")
        existing_roles = {
            str(value[0])
            for value in connection.execute(
                """
                SELECT coder_role FROM term_coding WHERE term_id = ?
                """,
                (term_id,),
            )
        }
        if role == "H2" and not {"AI", "H1"}.issubset(existing_roles):
            raise ValueError(
                "H2 term adjudication requires earlier independent AI and "
                f"H1 codes: {term_id}"
            )
        if role in {"AI", "H1"} and "H2" in existing_roles:
            raise ValueError(
                f"{role} term coding is frozen after H2 adjudication: "
                f"{term_id}"
            )
        domain = str(row.get("search_domain_label") or "").strip()
        definition = str(
            row.get("search_domain_definition") or ""
        ).strip()
        family = str(row.get("query_family_label") or "").strip()
        canonical_term = str(row.get("canonical_term") or "").strip()
        term_family = str(row.get("term_family_label") or "").strip()
        term_relation = str(row.get("term_relation") or "").strip().casefold()
        reason = str(row.get("reason") or "").strip()
        if decision == "include" and not all(
            (
                canonical_term,
                term_family,
                term_relation,
                domain,
                definition,
                family,
                reason,
            )
        ):
            raise ValueError(
                f"Included coding {term_id}/{role} lacks domain fields"
            )
        if decision == "include" and term_relation not in TERM_RELATIONS:
            raise ValueError(f"Invalid term_relation: {term_relation}")
        if not reason:
            raise ValueError(f"Coding {term_id}/{role} requires a reason")
        connection.execute(
            """
            INSERT INTO term_coding(
                term_id, coder_role, canonical_term, term_family_label,
                term_relation, search_domain_label,
                search_domain_definition, query_family_label,
                cross_domain, decision, reason, coded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(term_id, coder_role) DO UPDATE SET
                canonical_term = excluded.canonical_term,
                term_family_label = excluded.term_family_label,
                term_relation = excluded.term_relation,
                search_domain_label = excluded.search_domain_label,
                search_domain_definition = excluded.search_domain_definition,
                query_family_label = excluded.query_family_label,
                cross_domain = excluded.cross_domain,
                decision = excluded.decision,
                reason = excluded.reason,
                coded_at = excluded.coded_at
            """,
            (
                term_id,
                role,
                canonical_term,
                term_family,
                term_relation,
                domain,
                definition,
                family,
                int(parse_bool(row.get("cross_domain"), "cross_domain")),
                decision,
                reason,
                utc_now(),
            ),
        )
        imported += 1
    if imported:
        invalidate_stages(
            connection,
            SEARCH_FRAME_DOWNSTREAM_STAGES,
            "term coding changed",
        )
    connection.commit()
    return imported


def _coding_signature(row: sqlite3.Row) -> tuple[Any, ...]:
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


def _resolved_term_codes(
    connection: sqlite3.Connection,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    active_terms = connection.execute(
        """
        SELECT term_id, source_record_key, source_id, source_type,
               normalized_term, evidence_span
        FROM raw_terms
        WHERE status = 'active'
        ORDER BY term_id
        """
    ).fetchall()
    resolved: List[Dict[str, Any]] = []
    left: List[str] = []
    right: List[str] = []
    missing: List[str] = []
    for term in active_terms:
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
            missing.append(f"{term['term_id']}:AI/H1")
            continue
        ai = codes["AI"]
        h1 = codes["H1"]
        agreement = _coding_signature(ai) == _coding_signature(h1)
        left.append("|".join(map(str, _coding_signature(ai))))
        right.append("|".join(map(str, _coding_signature(h1))))
        both_exclude = (
            agreement
            and ai["decision"] == "exclude"
            and h1["decision"] == "exclude"
        )
        if both_exclude:
            final = h1
        else:
            final = codes.get("H2")
            if final is None:
                missing.append(f"{term['term_id']}:H2")
                continue
        resolved.append(
            {
                "term_id": term["term_id"],
                "source_record_key": term["source_record_key"],
                "source_id": term["source_id"],
                "source_type": term["source_type"],
                "normalized_term": term["normalized_term"],
                "evidence_span": term["evidence_span"],
                "decision": final["decision"],
                "canonical_term": final["canonical_term"],
                "term_family_label": final["term_family_label"],
                "term_relation": final["term_relation"],
                "search_domain_label": final["search_domain_label"],
                "search_domain_definition": final[
                    "search_domain_definition"
                ],
                "query_family_label": final["query_family_label"],
                "cross_domain": bool(final["cross_domain"]),
                "reason": final["reason"],
                "ai_h1_agreement": agreement,
            }
        )
    if missing:
        raise RuntimeError(
            "Term coding is incomplete: " + ", ".join(missing[:25])
        )
    metrics = {
        "n": len(left),
        "raw_agreement": raw_agreement(left, right) if left else None,
        "cohen_kappa": cohen_kappa(left, right) if left else None,
        "gwet_ac1": gwet_ac1(left, right) if left else None,
    }
    return resolved, metrics


def _domain_labels(value: str) -> List[str]:
    labels = [
        label.strip()
        for label in value.replace(";", "|").split("|")
        if label.strip()
    ]
    if not labels:
        raise ValueError("An included term must have a domain label")
    return sorted(set(labels), key=normalize_term)


def _split_term_block(
    domain_terms: Sequence[str],
    object_terms: Sequence[str],
    context_terms: Sequence[str],
    maximum_length: int = 1450,
) -> List[List[str]]:
    """Split only provider requests, never logical query semantics."""
    chunks: List[List[str]] = []
    current: List[str] = []
    for term in sorted(set(domain_terms), key=normalize_term):
        candidate = current + [term]
        expression = " AND ".join(
            (or_block(candidate), or_block(object_terms), or_block(context_terms))
        )
        if current and len(expression) > maximum_length:
            chunks.append(current)
            current = [term]
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _build_term_standardization(
    connection: sqlite3.Connection,
    included: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    """Persist raw-to-canonical mappings and synonym/abbreviation families."""
    family_rows: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in included:
        label = str(row["term_family_label"]).strip()
        if not label:
            raise RuntimeError(
                f"Included term lacks a term family: {row['term_id']}"
            )
        family_rows[label].append(row)
    connection.execute("DELETE FROM canonical_terms")
    connection.execute("DELETE FROM term_families")
    family_ids = {
        label: f"TF{index:04d}"
        for index, label in enumerate(
            sorted(family_rows, key=normalize_term),
            start=1,
        )
    }
    canonical_groups: Dict[
        tuple[str, str],
        List[Mapping[str, Any]],
    ] = defaultdict(list)
    canonical_display: Dict[tuple[str, str], str] = {}
    for label, members in family_rows.items():
        for row in members:
            canonical = str(row["canonical_term"]).strip()
            key = (label, normalize_term(canonical))
            canonical_groups[key].append(row)
            canonical_display.setdefault(key, canonical)
    canonical_ids = {
        key: f"CT{index:04d}"
        for index, key in enumerate(
            sorted(
                canonical_groups,
                key=lambda value: (
                    normalize_term(value[0]),
                    value[1],
                ),
            ),
            start=1,
        )
    }
    raw_sources = {
        str(row["term_id"]): str(row["source_id"])
        for row in connection.execute(
            "SELECT term_id, source_id FROM raw_terms"
        )
    }
    for label in sorted(family_rows, key=normalize_term):
        members = family_rows[label]
        canonical_term_ids = sorted(
            canonical_ids[key]
            for key in canonical_groups
            if key[0] == label
        )
        raw_term_ids = sorted({str(row["term_id"]) for row in members})
        source_ids = sorted(
            {
                raw_sources.get(term_id, "")
                for term_id in raw_term_ids
                if raw_sources.get(term_id, "")
            }
        )
        definitions = sorted(
            {
                str(row["search_domain_definition"]).strip()
                for row in members
                if str(row["search_domain_definition"]).strip()
            },
            key=normalize_term,
        )
        connection.execute(
            """
            INSERT INTO term_families(
                term_family_id, label, construct_definition,
                canonical_term_ids_json, raw_term_ids_json,
                source_ids_json, status
            ) VALUES (?, ?, ?, ?, ?, ?, 'active')
            """,
            (
                family_ids[label],
                label,
                " | ".join(definitions),
                json.dumps(canonical_term_ids, ensure_ascii=False),
                json.dumps(raw_term_ids, ensure_ascii=False),
                json.dumps(source_ids, ensure_ascii=False),
            ),
        )
    for key in sorted(
        canonical_groups,
        key=lambda value: (normalize_term(value[0]), value[1]),
    ):
        label, _ = key
        members = canonical_groups[key]
        raw_term_ids = sorted({str(row["term_id"]) for row in members})
        relation_map = {
            str(row["term_id"]): str(row["term_relation"])
            for row in sorted(members, key=lambda value: str(value["term_id"]))
        }
        source_ids = sorted(
            {
                raw_sources.get(term_id, "")
                for term_id in raw_term_ids
                if raw_sources.get(term_id, "")
            }
        )
        connection.execute(
            """
            INSERT INTO canonical_terms(
                canonical_term_id, term_family_id, canonical_term,
                raw_term_ids_json, relation_map_json, source_ids_json,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, 'active')
            """,
            (
                canonical_ids[key],
                family_ids[label],
                canonical_display[key],
                json.dumps(raw_term_ids, ensure_ascii=False),
                json.dumps(relation_map, ensure_ascii=False, sort_keys=True),
                json.dumps(source_ids, ensure_ascii=False),
            ),
        )
    return {
        "canonical_terms": len(canonical_groups),
        "term_families": len(family_rows),
    }


def derive_search_frame(
    connection: sqlite3.Connection,
) -> Dict[str, Any]:
    """Derive K/Q/P from fully adjudicated evidence-linked terms."""
    _assert_not_frozen(connection)
    require_complete(connection, ["bootstrap_retrieval_complete"])
    if connection.execute(
        """
        SELECT COUNT(*) FROM discovery_queries
        WHERE status IN ('active', 'network')
        """
    ).fetchone()[0]:
        required_zero = int(
            read_json(SATURATION_PROTOCOL_PATH)["sequential_review"][
                "minimum_consecutive_zero_novelty_rounds"
            ]
        )
        frozen = connection.execute(
            """
            SELECT 1 FROM discovery_review_rounds
            WHERE saturation_phase = 'search_frame_discovery'
              AND reviewer_role = 'H2'
              AND decision = 'freeze'
              AND fully_reviewed = 1
              AND consecutive_zero_rounds >= ?
              AND new_nonredundant_english_terms = 0
              AND new_canonical_indicator_families = 0
              AND iteration = (
                  SELECT MAX(iteration)
                  FROM discovery_review_rounds
                  WHERE saturation_phase = 'search_frame_discovery'
              )
            LIMIT 1
            """,
            (required_zero,),
        ).fetchone()
        if frozen is None:
            raise RuntimeError(
                "Search-frame derivation requires H2-approved "
                f"{required_zero}-round dual-zero discovery saturation"
            )
    resolved, metrics = _resolved_term_codes(connection)
    included = [row for row in resolved if row["decision"] == "include"]
    if not included:
        raise RuntimeError("No terms survived adjudicated coding")
    frame_version = int(
        connection.execute(
            "SELECT COALESCE(MAX(frame_version), 0) + 1 "
            "FROM search_frame_versions"
        ).fetchone()[0]
    )
    standardization_counts = _build_term_standardization(
        connection,
        included,
    )
    domain_members: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    domain_definitions: Dict[str, set[str]] = defaultdict(set)
    family_members: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(
        list
    )
    for row in included:
        labels = _domain_labels(row["search_domain_label"])
        if len(labels) > 1 and not row["cross_domain"]:
            raise RuntimeError(
                f"Multi-domain term lacks cross_domain approval: "
                f"{row['term_id']}"
            )
        for domain_label in labels:
            domain_members[domain_label].append(row)
            domain_definitions[domain_label].add(
                row["search_domain_definition"].strip()
            )
            family_members[
                (domain_label, row["query_family_label"].strip())
            ].append(row)
    hint_only = [
        label
        for label, members in domain_members.items()
        if all(
            member["source_type"] in NONAUTHORIZING_TERM_SOURCE_TYPES
            for member in members
        )
    ]
    if hint_only:
        raise RuntimeError(
            "Pilot or derived-hint terms cannot independently establish "
            "domains: "
            + ", ".join(sorted(hint_only))
        )
    hint_only_query_families = [
        f"{domain_label}::{family_label}"
        for (domain_label, family_label), members in family_members.items()
        if all(
            member["source_type"] in NONAUTHORIZING_TERM_SOURCE_TYPES
            for member in members
        )
    ]
    if hint_only_query_families:
        raise RuntimeError(
            "Pilot or derived-hint terms cannot independently establish "
            "logical query families: "
            + ", ".join(sorted(hint_only_query_families))
        )
    connection.execute(
        """
        DELETE FROM query_runs
        WHERE physical_query_id IN (
            SELECT p.physical_query_id
            FROM physical_queries p
            JOIN logical_queries l USING(logical_query_id)
            WHERE l.logical_query_id LIKE 'L%'
        )
        """
    )
    connection.execute(
        """
        DELETE FROM query_hits
        WHERE physical_query_id IN (
            SELECT p.physical_query_id
            FROM physical_queries p
            JOIN logical_queries l USING(logical_query_id)
            WHERE l.logical_query_id LIKE 'L%'
        )
        """
    )
    connection.execute(
        """
        DELETE FROM physical_queries
        WHERE logical_query_id LIKE 'L%'
        """
    )
    connection.execute(
        """
        DELETE FROM press_reviews
        WHERE logical_query_id LIKE 'L%'
        """
    )
    connection.execute(
        """
        DELETE FROM logical_queries
        WHERE logical_query_id LIKE 'L%'
        """
    )
    connection.execute("DELETE FROM search_domains")
    protocol = read_json(PROTOCOL_PATH)
    object_terms = [str(value) for value in protocol["bootstrap"]["object_terms"]]
    context_terms = [
        str(value) for value in protocol["bootstrap"]["evidence_terms"]
    ]
    domain_id_by_label: Dict[str, str] = {}
    for index, label in enumerate(
        sorted(domain_members, key=normalize_term),
        start=1,
    ):
        domain_id = f"SD{index:03d}"
        domain_id_by_label[label] = domain_id
        definitions = sorted(domain_definitions[label], key=normalize_term)
        definition = " | ".join(definitions)
        term_ids = sorted(
            {str(row["term_id"]) for row in domain_members[label]}
        )
        connection.execute(
            """
            INSERT INTO search_domains(
                search_domain_id, label, definition, term_ids_json,
                status, h2_approved, decision_reason
            ) VALUES (?, ?, ?, ?, 'active', 1,
                      'H2 term-level domain assignment approval')
            """,
            (
                domain_id,
                label,
                definition,
                json.dumps(term_ids, ensure_ascii=False),
            ),
        )
    logical_count = 0
    physical_count = 0
    seen_semantics: Dict[str, str] = {}
    for (domain_label, family_label), members in sorted(
        family_members.items(),
        key=lambda item: (
            normalize_term(item[0][0]),
            normalize_term(item[0][1]),
        ),
    ):
        terms = sorted(
            {str(row["normalized_term"]) for row in members},
            key=normalize_term,
        )
        semantic_hash = json_hash(
            {
                "domain": normalize_term(domain_label),
                "family": normalize_term(family_label),
                "domain_terms": terms,
                "object_terms": sorted(object_terms),
                "context_terms": sorted(context_terms),
            }
        )
        logical_count += 1
        logical_id = f"L{logical_count:04d}"
        logical_expression = " AND ".join(
            (or_block(terms), or_block(object_terms), or_block(context_terms))
        )
        status = "active"
        archive_reason = ""
        if semantic_hash in seen_semantics:
            status = "archived"
            archive_reason = (
                "R_LOGICAL_DUPLICATE_OF_" + seen_semantics[semantic_hash]
            )
        else:
            seen_semantics[semantic_hash] = logical_id
        logical_hash = json_hash(
            {
                "logical_query_id": logical_id,
                "domain": domain_label,
                "family": family_label,
                "expression": logical_expression,
            }
        )
        connection.execute(
            """
            INSERT INTO logical_queries(
                logical_query_id, query_version, search_domain_id,
                family_label, logical_expression, object_terms_json,
                domain_terms_json, context_terms_json, status,
                archive_reason, press_status, query_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                logical_id,
                frame_version,
                domain_id_by_label[domain_label],
                family_label,
                logical_expression,
                json.dumps(object_terms, ensure_ascii=False),
                json.dumps(terms, ensure_ascii=False),
                json.dumps(context_terms, ensure_ascii=False),
                status,
                archive_reason,
                logical_hash,
            ),
        )
        for part_index, chunk in enumerate(
            _split_term_block(terms, object_terms, context_terms),
            start=1,
        ):
            physical_count += 1
            physical_id = f"{logical_id}__P{part_index:03d}"
            expression = " AND ".join(
                (
                    or_block(chunk),
                    or_block(object_terms),
                    or_block(context_terms),
                )
            )
            filter_values = [
                f"to_publication_date:{protocol['cutoff_date']}",
                "type:article|review",
            ]
            if protocol.get("from_date"):
                filter_values.insert(
                    0,
                    f"from_publication_date:{protocol['from_date']}",
                )
            filter_expression = ",".join(filter_values)
            connection.execute(
                """
                INSERT INTO physical_queries(
                    physical_query_id, logical_query_id, provider,
                    expression, filter_expression, status, query_hash
                ) VALUES (?, ?, 'OpenAlex', ?, ?, ?, ?)
                """,
                (
                    physical_id,
                    logical_id,
                    expression,
                    filter_expression,
                    status,
                    query_definition_hash(expression, filter_expression),
                ),
            )
    active_k = connection.execute(
        "SELECT COUNT(*) FROM search_domains WHERE status = 'active'"
    ).fetchone()[0]
    active_q = connection.execute(
        """
        SELECT COUNT(*) FROM logical_queries
        WHERE status = 'active' AND logical_query_id LIKE 'L%'
        """
    ).fetchone()[0]
    active_p = connection.execute(
        """
        SELECT COUNT(*) FROM physical_queries p
        JOIN logical_queries l ON l.logical_query_id = p.logical_query_id
        WHERE p.status = 'active' AND l.status = 'active'
          AND l.logical_query_id LIKE 'L%'
        """
    ).fetchone()[0]
    frame_body = {
        "frame_version": frame_version,
        "input_terms": [
            {
                key: row[key]
                for key in (
                    "term_id",
                    "source_record_key",
                    "source_id",
                    "source_type",
                    "evidence_span",
                    "canonical_term",
                    "term_family_label",
                    "term_relation",
                    "search_domain_label",
                    "query_family_label",
                    "decision",
                )
            }
            for row in resolved
        ],
        "domains": [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM search_domains ORDER BY search_domain_id"
            )
        ],
        "logical_queries": [
            dict(row)
            for row in connection.execute(
                """
                SELECT * FROM logical_queries
                WHERE logical_query_id LIKE 'L%'
                ORDER BY logical_query_id
                """
            )
        ],
        "physical_queries": [
            dict(row)
            for row in connection.execute(
                """
                SELECT * FROM physical_queries
                WHERE logical_query_id LIKE 'L%'
                ORDER BY physical_query_id
                """
            )
        ],
    }
    input_term_hash = json_hash({"terms": frame_body["input_terms"]})
    frame_hash = json_hash(frame_body)
    connection.execute(
        """
        UPDATE search_frame_versions
        SET status = 'superseded'
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
            frame_version,
            input_term_hash,
            frame_hash,
            json.dumps(
                {"K": active_k, "Q": active_q, "P": active_p},
                sort_keys=True,
            ),
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
        "terms_coded",
        "complete",
        {
            "active_terms": len(resolved),
            "included_terms": len(included),
            "agreement": metrics,
            **standardization_counts,
        },
    )
    set_stage(
        connection,
        "search_frame_derived",
        "complete",
        {
            "frame_version": frame_version,
            "K": active_k,
            "Q": active_q,
            "P": active_p,
            "frame_hash": frame_hash,
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
        "search frame re-derived",
    )
    connection.commit()
    return {
        "frame_version": frame_version,
        "K": active_k,
        "Q": active_q,
        "P": active_p,
        "frame_hash": frame_hash,
        "agreement": metrics,
    }


def export_press(
    connection: sqlite3.Connection,
    output_path: Path,
) -> int:
    """Export active logical queries for independent PRESS review."""
    require_complete(connection, ["search_frame_derived"])
    rows: List[Dict[str, Any]] = []
    for row in connection.execute(
        """
        SELECT logical_query_id, search_domain_id, family_label,
               logical_expression
        FROM logical_queries
        WHERE status = 'active' AND logical_query_id LIKE 'L%'
        ORDER BY logical_query_id
        """
    ):
        rows.append(
            {
                **dict(row),
                "reviewer_role": "H2",
                "concepts_complete": "",
                "boolean_logic_valid": "",
                "spelling_valid": "",
                "phrases_valid": "",
                "limits_justified": "",
                "covered_by_logical_query_id": "",
                "logical_coverage_verified": "",
                "result_set_coverage_verified": "",
                "independent_construct_role": "",
                "decision": "",
                "notes": "",
            }
        )
    write_csv(output_path, rows, PRESS_FIELDS)
    return len(rows)


def import_press(
    connection: sqlite3.Connection,
    input_path: Path,
) -> Dict[str, int]:
    """Import PRESS decisions and archive reviewed redundancies."""
    _assert_not_frozen(connection)
    snapshot_path = snapshot_import_file(
        connection,
        input_path,
        "press",
    )
    rows = read_csv(snapshot_path)
    counts = {"pass": 0, "revise": 0, "archive_redundant": 0}
    for row in rows:
        logical_id = str(row.get("logical_query_id") or "").strip()
        decision = str(row.get("decision") or "").strip().casefold()
        if not logical_id or not decision:
            continue
        if str(row.get("reviewer_role") or "").strip().upper() != "H2":
            raise ValueError("PRESS reviewer_role must be H2")
        if decision not in counts:
            raise ValueError(f"Invalid PRESS decision: {decision}")
        checklist = {
            field: parse_bool(row.get(field), field)
            for field in (
                "concepts_complete",
                "boolean_logic_valid",
                "spelling_valid",
                "phrases_valid",
                "limits_justified",
            )
        }
        notes = str(row.get("notes") or "").strip()
        if decision == "pass" and not all(checklist.values()):
            raise ValueError(
                f"PRESS pass requires all checks: {logical_id}"
            )
        if decision != "pass" and not notes:
            raise ValueError(
                f"Non-pass PRESS decision requires notes: {logical_id}"
            )
        covered_by = str(
            row.get("covered_by_logical_query_id") or ""
        ).strip()
        if decision == "archive_redundant":
            logical_coverage = parse_bool(
                row.get("logical_coverage_verified"),
                "logical_coverage_verified",
            )
            result_coverage = parse_bool(
                row.get("result_set_coverage_verified") or "false",
                "result_set_coverage_verified",
            )
            independent_role = parse_bool(
                row.get("independent_construct_role"),
                "independent_construct_role",
            )
            if (
                not covered_by
                or covered_by == logical_id
                or not logical_coverage
                or independent_role
            ):
                raise ValueError(
                    "Redundancy archival requires a different covering "
                    "query, verified logical coverage, and no independent "
                    "construct role. Result-set coverage is computed by "
                    "the validation command."
                )
            result_coverage = False
        else:
            logical_coverage = False
            result_coverage = False
            independent_role = True
        exists = connection.execute(
            """
            SELECT 1 FROM logical_queries
            WHERE logical_query_id = ? AND logical_query_id LIKE 'L%'
            """,
            (logical_id,),
        ).fetchone()
        if exists is None:
            raise ValueError(f"Unknown formal logical query: {logical_id}")
        if decision == "archive_redundant":
            covering_exists = connection.execute(
                """
                SELECT 1 FROM logical_queries
                WHERE logical_query_id = ? AND status = 'active'
                  AND logical_query_id LIKE 'L%'
                """,
                (covered_by,),
            ).fetchone()
            if covering_exists is None:
                raise ValueError(
                    f"Covering query is not active: {covered_by}"
                )
        connection.execute(
            """
            INSERT INTO press_reviews(
                logical_query_id, reviewer_role, concepts_complete,
                boolean_logic_valid, spelling_valid, phrases_valid,
                limits_justified, covered_by_logical_query_id,
                logical_coverage_verified, result_set_coverage_verified,
                independent_construct_role, decision, notes, reviewed_at
            ) VALUES (?, 'H2', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(logical_query_id) DO UPDATE SET
                reviewer_role = 'H2',
                concepts_complete = excluded.concepts_complete,
                boolean_logic_valid = excluded.boolean_logic_valid,
                spelling_valid = excluded.spelling_valid,
                phrases_valid = excluded.phrases_valid,
                limits_justified = excluded.limits_justified,
                covered_by_logical_query_id =
                    excluded.covered_by_logical_query_id,
                logical_coverage_verified =
                    excluded.logical_coverage_verified,
                result_set_coverage_verified =
                    excluded.result_set_coverage_verified,
                independent_construct_role =
                    excluded.independent_construct_role,
                decision = excluded.decision,
                notes = excluded.notes,
                reviewed_at = excluded.reviewed_at
            """,
            (
                logical_id,
                *(int(value) for value in checklist.values()),
                covered_by,
                int(logical_coverage),
                int(result_coverage),
                int(independent_role),
                decision,
                notes,
                utc_now(),
            ),
        )
        if decision == "archive_redundant":
            connection.execute(
                """
                UPDATE logical_queries
                SET press_status = 'redundancy_pending',
                    press_reviewer = 'H2', press_notes = ?
                WHERE logical_query_id = ?
                """,
                (notes, logical_id),
            )
        else:
            connection.execute(
                """
                UPDATE logical_queries
                SET press_status = ?, press_reviewer = 'H2',
                    press_notes = ?
                WHERE logical_query_id = ?
                """,
                (decision, notes, logical_id),
            )
        counts[decision] += 1
    _refresh_search_domain_status(connection)
    if sum(counts.values()):
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
            "PRESS decisions changed",
        )
    connection.commit()
    return counts


def import_hidden_seeds(
    connection: sqlite3.Connection,
    input_path: Path,
) -> int:
    """Register H2-supplied seeds without exposing them to term generation."""
    _assert_not_frozen(connection)
    snapshot_path = snapshot_import_file(
        connection,
        input_path,
        "hidden_seeds",
    )
    rows = read_csv(snapshot_path)
    imported = 0
    for row in rows:
        if not str(row.get("doi") or "").strip():
            continue
        role = str(row.get("seed_role") or "validation").casefold()
        supplied_by = str(row.get("supplied_by") or "").strip().upper()
        hidden = parse_bool(
            row.get("hidden_during_development") or "true",
            "hidden_during_development",
        )
        if role != "validation" or supplied_by != "H2" or not hidden:
            raise ValueError(
                "Hidden seeds must be validation/H2/hidden=true"
            )
        seed_id = str(row.get("seed_id") or "").strip()
        doi = normalize_doi(row.get("doi"))
        if not seed_id:
            seed_id = "VAL_" + sha256_bytes(doi.encode("utf-8"))[:12]
        language = str(row.get("language") or "en").strip().casefold()
        eligibility = str(
            row.get("eligibility_status") or "eligible"
        ).strip().casefold()
        if eligibility not in {"eligible", "ineligible"}:
            raise ValueError(
                f"Invalid hidden-seed eligibility: {eligibility}"
            )
        citation = str(row.get("citation") or "").strip()
        publication_year = (
            int(row["publication_year"])
            if str(row.get("publication_year") or "").strip()
            else None
        )
        if eligibility == "eligible" and (
            language != "en"
            or not citation
            or publication_year is None
        ):
            raise ValueError(
                "An eligible hidden seed requires English language, a "
                "traceable citation, and publication_year"
            )
        connection.execute(
            """
            INSERT INTO evidence_seeds(
                seed_id, doi, citation, publication_year, language,
                seed_role, supplied_by, hidden_during_development,
                eligibility_status, nonrecall_reason
            ) VALUES (?, ?, ?, ?, ?, 'validation', 'H2', 1, ?, ?)
            ON CONFLICT(seed_id) DO UPDATE SET
                doi = excluded.doi,
                citation = excluded.citation,
                publication_year = excluded.publication_year,
                language = excluded.language,
                eligibility_status = excluded.eligibility_status,
                nonrecall_reason = excluded.nonrecall_reason
            """,
            (
                seed_id,
                doi,
                citation,
                publication_year,
                language,
                eligibility,
                str(row.get("nonrecall_reason") or "").strip(),
            ),
        )
        imported += 1
    if imported:
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
            "hidden validation seed set changed",
        )
    connection.commit()
    return imported


def export_seed_template(output_path: Path) -> None:
    """Write an empty hidden-validation seed template."""
    write_csv(output_path, [], SEED_FIELDS)


def export_hidden_seed_search_log_template(output_path: Path) -> int:
    """Write the three required H2 hidden-seed provenance routes."""
    rows = [
        {
            "search_run_id": f"H2_HIDDEN_SEED_{index:02d}",
            "reviewer_role": "H2",
            "route": route,
            "source_name": "",
            "exact_query_or_seed": "",
            "executed_at": "",
            "retrieved_count": "",
            "screened_count": "",
            "eligible_seed_count": "",
            "eligible_seed_dois": "",
            "completion_status": "",
            "notes": "",
        }
        for index, route in enumerate(
            sorted(HIDDEN_SEED_SEARCH_ROUTES),
            start=1,
        )
    ]
    write_csv(output_path, rows, HIDDEN_SEED_SEARCH_LOG_FIELDS)
    return len(rows)


def hidden_seed_search_log_status(
    connection: sqlite3.Connection,
) -> Dict[str, Any]:
    """Summarize reviewed H2 seed-discovery route documentation."""
    complete_routes = {
        str(row["route"])
        for row in connection.execute(
            """
            SELECT DISTINCT route
            FROM hidden_seed_search_log
            WHERE reviewer_role = 'H2'
              AND completion_status = 'complete'
            """
        )
    }
    return {
        "required_routes": sorted(HIDDEN_SEED_SEARCH_ROUTES),
        "complete_routes": sorted(complete_routes),
        "missing_routes": sorted(
            HIDDEN_SEED_SEARCH_ROUTES - complete_routes
        ),
        "complete_runs": int(
            connection.execute(
                """
                SELECT COUNT(*) FROM hidden_seed_search_log
                WHERE reviewer_role = 'H2'
                  AND completion_status = 'complete'
                """
            ).fetchone()[0]
        ),
    }


def import_hidden_seed_search_log(
    connection: sqlite3.Connection,
    input_path: Path,
) -> Dict[str, Any]:
    """Import H2's reviewed search and citation-tracing log."""
    _assert_not_frozen(connection)
    snapshot_path = snapshot_import_file(
        connection,
        input_path,
        "hidden_seed_search_log",
    )
    imported = 0
    for row in read_csv(snapshot_path):
        route = str(row.get("route") or "").strip().casefold()
        if not route:
            continue
        role = str(row.get("reviewer_role") or "").strip().upper()
        if role != "H2":
            raise ValueError("Hidden-seed search logging is restricted to H2")
        if route not in HIDDEN_SEED_SEARCH_ROUTES:
            raise ValueError(f"Invalid hidden-seed search route: {route}")
        source_name = str(row.get("source_name") or "").strip()
        exact_query_or_seed = str(
            row.get("exact_query_or_seed") or ""
        ).strip()
        executed_at = str(row.get("executed_at") or "").strip()
        completion_status = str(
            row.get("completion_status") or ""
        ).strip().casefold()
        notes = str(row.get("notes") or "").strip()
        if not all(
            (
                source_name,
                exact_query_or_seed,
                executed_at,
                completion_status,
                notes,
            )
        ):
            raise ValueError(
                f"Hidden-seed search row lacks provenance fields: {route}"
            )
        try:
            executed_datetime = datetime.fromisoformat(
                executed_at.replace("Z", "+00:00")
            )
        except ValueError as error:
            raise ValueError(
                f"Invalid ISO date/time for hidden-seed search: {executed_at}"
            ) from error
        if executed_datetime.tzinfo is None:
            raise ValueError(
                "Hidden-seed search execution time requires a UTC offset"
            )
        if completion_status not in {"complete", "pending"}:
            raise ValueError(
                "completion_status must be complete or pending"
            )
        counts: Dict[str, int] = {}
        for field in (
            "retrieved_count",
            "screened_count",
            "eligible_seed_count",
        ):
            raw_value = str(row.get(field) or "").strip()
            try:
                counts[field] = int(raw_value)
            except ValueError as error:
                raise ValueError(
                    f"{field} must be a non-negative integer"
                ) from error
            if counts[field] < 0:
                raise ValueError(
                    f"{field} must be a non-negative integer"
                )
        if counts["screened_count"] > counts["retrieved_count"]:
            raise ValueError(
                "screened_count cannot exceed retrieved_count"
            )
        if counts["eligible_seed_count"] > counts["screened_count"]:
            raise ValueError(
                "eligible_seed_count cannot exceed screened_count"
            )
        eligible_dois: List[str] = []
        for raw_doi in re.split(
            r"[;|\r\n]+",
            str(row.get("eligible_seed_dois") or ""),
        ):
            doi = normalize_doi(raw_doi)
            if not doi:
                continue
            if not re.fullmatch(r"10\.\d{4,9}/\S+", doi):
                raise ValueError(
                    f"Invalid DOI in eligible_seed_dois: {raw_doi}"
                )
            eligible_dois.append(doi)
        eligible_dois = sorted(set(eligible_dois))
        if len(eligible_dois) != counts["eligible_seed_count"]:
            raise ValueError(
                "eligible_seed_count must equal the number of distinct "
                "DOIs listed in eligible_seed_dois"
            )
        eligible_dois_json = json.dumps(
            eligible_dois,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        search_run_id = str(row.get("search_run_id") or "").strip()
        if not search_run_id:
            identity = "|".join(
                (role, route, source_name, exact_query_or_seed, executed_at)
            )
            search_run_id = (
                "H2SEARCH_"
                + sha256_bytes(identity.encode("utf-8"))[:16].upper()
            )
        connection.execute(
            """
            INSERT INTO hidden_seed_search_log(
                search_run_id, reviewer_role, route, source_name,
                exact_query_or_seed, executed_at, retrieved_count,
                screened_count, eligible_seed_count,
                eligible_seed_dois_json, completion_status, notes,
                imported_at
            ) VALUES (?, 'H2', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(search_run_id) DO UPDATE SET
                reviewer_role = excluded.reviewer_role,
                route = excluded.route,
                source_name = excluded.source_name,
                exact_query_or_seed = excluded.exact_query_or_seed,
                executed_at = excluded.executed_at,
                retrieved_count = excluded.retrieved_count,
                screened_count = excluded.screened_count,
                eligible_seed_count = excluded.eligible_seed_count,
                eligible_seed_dois_json =
                    excluded.eligible_seed_dois_json,
                completion_status = excluded.completion_status,
                notes = excluded.notes,
                imported_at = excluded.imported_at
            """,
            (
                search_run_id,
                route,
                source_name,
                exact_query_or_seed,
                executed_at,
                counts["retrieved_count"],
                counts["screened_count"],
                counts["eligible_seed_count"],
                eligible_dois_json,
                completion_status,
                notes,
                utc_now(),
            ),
        )
        imported += 1
    if imported:
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
            "hidden validation seed search log changed",
        )
    connection.commit()
    return {
        "imported": imported,
        **hidden_seed_search_log_status(connection),
    }


def export_seed_supplement_template(
    connection: sqlite3.Connection,
    output_path: Path,
) -> int:
    """Export OpenAlex-unindexable eligible seeds for direct inclusion."""
    rows = []
    for seed in connection.execute(
        """
        SELECT * FROM evidence_seeds
        WHERE eligibility_status = 'eligible'
          AND language = 'en'
          AND recall_status = 'supplement_required'
        ORDER BY seed_role, seed_id
        """
    ):
        rows.append(
            {
                "seed_id": seed["seed_id"],
                "doi": seed["doi"],
                "title": "",
                "abstract": "",
                "publication_year": seed["publication_year"] or "",
                "work_type": "article",
                "language": "en",
                "source_url": "",
                "supplied_by": "H2",
            }
        )
    write_csv(output_path, rows, SEED_SUPPLEMENT_FIELDS)
    return len(rows)


def import_seed_supplements(
    connection: sqlite3.Connection,
    input_path: Path,
) -> int:
    """Directly include eligible English seeds absent from OpenAlex."""
    _assert_not_frozen(connection)
    snapshot_path = snapshot_import_file(
        connection,
        input_path,
        "seed_supplements",
    )
    rows = read_csv(snapshot_path)
    imported = 0
    for row in rows:
        seed_id = str(row.get("seed_id") or "").strip()
        if not seed_id:
            continue
        seed = connection.execute(
            """
            SELECT * FROM evidence_seeds
            WHERE seed_id = ? AND recall_status = 'supplement_required'
            """,
            (seed_id,),
        ).fetchone()
        if seed is None:
            raise ValueError(
                f"Seed is not awaiting an OpenAlex supplement: {seed_id}"
            )
        doi = normalize_doi(row.get("doi"))
        if doi != seed["doi"]:
            raise ValueError(f"Supplement DOI mismatch for {seed_id}")
        if str(row.get("language") or "").strip().casefold() != "en":
            raise ValueError("Only English seed supplements are eligible")
        if str(row.get("supplied_by") or "").strip().upper() != "H2":
            raise ValueError("Seed supplements require H2 confirmation")
        title = str(row.get("title") or "").strip()
        abstract = str(row.get("abstract") or "").strip()
        source_url = str(row.get("source_url") or "").strip()
        work_type = str(row.get("work_type") or "").strip().casefold()
        if not all((title, abstract, source_url)):
            raise ValueError(
                f"Supplement requires English title, abstract, and source: "
                f"{seed_id}"
            )
        if work_type not in {"article", "review"}:
            raise ValueError(f"Invalid supplement work type: {work_type}")
        publication_year = int(
            _required_seed_value(row, "publication_year", seed_id)
        )
        record_key = f"doi:{doi}"
        raw = {
            "source": "H2_seed_supplement",
            "seed_id": seed_id,
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "publication_year": publication_year,
            "work_type": work_type,
            "language": "en",
            "source_url": source_url,
        }
        connection.execute(
            """
            INSERT OR IGNORE INTO records(
                provider, record_key, provider_id, doi, title, abstract,
                language, publication_year, work_type, source_url,
                referenced_works_json, raw_json, retrieval_route,
                first_seen_at
            ) VALUES (
                'Manual', ?, ?, ?, ?, ?, 'en', ?, ?, ?, '[]', ?,
                'manual_seed_supplement', ?
            )
            """,
            (
                record_key,
                f"manual:{seed_id}",
                doi,
                title,
                abstract,
                publication_year,
                work_type,
                source_url,
                json.dumps(raw, ensure_ascii=False, sort_keys=True),
                utc_now(),
            ),
        )
        connection.execute(
            """
            UPDATE evidence_seeds
            SET recall_status = 'supplemented',
                nonrecall_reason = 'OPENALEX_NOT_INDEXABLE_DIRECT_INCLUDED'
            WHERE seed_id = ?
            """,
            (seed_id,),
        )
        imported += 1
    connection.commit()
    return imported


def _required_seed_value(
    row: Mapping[str, Any],
    field: str,
    seed_id: str,
) -> str:
    value = str(row.get(field) or "").strip()
    if not value:
        raise ValueError(f"Seed supplement {seed_id} requires {field}")
    return value


def _archive_zero_hit_families(
    connection: sqlite3.Connection,
    totals: Mapping[str, int],
) -> List[str]:
    archived: List[str] = []
    logical_ids = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT logical_query_id
            FROM logical_queries
            WHERE status = 'active' AND logical_query_id LIKE 'L%'
            ORDER BY logical_query_id
            """
        )
    ]
    for logical_id in logical_ids:
        physical_ids = [
            str(row[0])
            for row in connection.execute(
                """
                SELECT physical_query_id FROM physical_queries
                WHERE logical_query_id = ? AND status = 'active'
                ORDER BY physical_query_id
                """,
                (logical_id,),
            )
        ]
        if physical_ids and all(totals.get(value, -1) == 0 for value in physical_ids):
            connection.execute(
                """
                UPDATE logical_queries
                SET status = 'archived', archive_reason = 'R_ZERO_HIT'
                WHERE logical_query_id = ?
                """,
                (logical_id,),
            )
            connection.execute(
                """
                UPDATE physical_queries SET status = 'archived'
                WHERE logical_query_id = ?
                """,
                (logical_id,),
            )
            archived.append(logical_id)
    _refresh_search_domain_status(connection)
    return archived


def _logical_physical_ids(
    connection: sqlite3.Connection,
    logical_query_id: str,
) -> List[str]:
    return [
        str(row[0])
        for row in connection.execute(
            """
            SELECT physical_query_id FROM physical_queries
            WHERE logical_query_id = ? AND status = 'active'
            ORDER BY physical_query_id
            """,
            (logical_query_id,),
        )
    ]


def _resolve_redundancy_reviews(
    connection: sqlite3.Connection,
    fetcher: Any | None = None,
) -> List[str]:
    """Verify claimed query redundancy against complete provider hit sets."""
    reviews = connection.execute(
        """
        SELECT * FROM press_reviews
        WHERE decision = 'archive_redundant'
          AND result_set_coverage_verified = 0
        ORDER BY logical_query_id
        """
    ).fetchall()
    archived: List[str] = []
    for review in reviews:
        candidate_id = str(review["logical_query_id"])
        covering_id = str(review["covered_by_logical_query_id"])
        covering_review = connection.execute(
            """
            SELECT decision FROM press_reviews
            WHERE logical_query_id = ?
            """,
            (covering_id,),
        ).fetchone()
        if (
            covering_review is not None
            and covering_review["decision"] == "archive_redundant"
        ):
            raise RuntimeError(
                "Redundancy chains are not allowed; H2 must identify a "
                f"surviving covering query for {candidate_id}"
            )
        candidate_physical = _logical_physical_ids(
            connection,
            candidate_id,
        )
        covering_physical = _logical_physical_ids(
            connection,
            covering_id,
        )
        if not candidate_physical or not covering_physical:
            raise RuntimeError(
                f"Redundancy pair lacks active physical queries: "
                f"{candidate_id}/{covering_id}"
            )
        for physical_id in sorted(
            set(candidate_physical + covering_physical)
        ):
            kwargs: Dict[str, Any] = {}
            if fetcher is not None:
                kwargs["fetcher"] = fetcher
            retrieve_physical_query(
                connection,
                physical_id,
                "redundancy_validation",
                **kwargs,
            )
        candidate_placeholders = ",".join(
            "?" for _ in candidate_physical
        )
        covering_placeholders = ",".join(
            "?" for _ in covering_physical
        )
        uncovered = connection.execute(
            f"""
            SELECT record_key FROM query_hits
            WHERE run_role = 'redundancy_validation'
              AND physical_query_id IN ({candidate_placeholders})
            EXCEPT
            SELECT record_key FROM query_hits
            WHERE run_role = 'redundancy_validation'
              AND physical_query_id IN ({covering_placeholders})
            LIMIT 1
            """,
            tuple(candidate_physical + covering_physical),
        ).fetchone()
        if uncovered is not None:
            connection.execute(
                """
                UPDATE logical_queries
                SET press_status = 'redundancy_rejected',
                    press_notes = press_notes ||
                        ' | result-set subset check failed'
                WHERE logical_query_id = ?
                """,
                (candidate_id,),
            )
            connection.commit()
            raise RuntimeError(
                f"Claimed redundant query has unique results: {candidate_id}"
            )
        connection.execute(
            """
            UPDATE press_reviews
            SET result_set_coverage_verified = 1,
                notes = notes || ' | complete result-set subset verified'
            WHERE logical_query_id = ?
            """,
            (candidate_id,),
        )
        connection.execute(
            """
            UPDATE logical_queries
            SET status = 'archived',
                archive_reason = ?,
                press_status = 'archived_redundant'
            WHERE logical_query_id = ?
            """,
            (
                f"R_PRESS_REDUNDANT_OF_{covering_id}",
                candidate_id,
            ),
        )
        connection.execute(
            """
            UPDATE physical_queries SET status = 'archived'
            WHERE logical_query_id = ?
            """,
            (candidate_id,),
        )
        archived.append(candidate_id)
        connection.commit()
    _refresh_search_domain_status(connection)
    return archived


def _refresh_search_domain_status(
    connection: sqlite3.Connection,
) -> None:
    """Archive domains with no surviving non-redundant logical query."""
    connection.execute(
        """
        UPDATE search_domains
        SET status = CASE
            WHEN EXISTS (
                SELECT 1 FROM logical_queries l
                WHERE l.search_domain_id = search_domains.search_domain_id
                  AND l.status = 'active'
                  AND l.logical_query_id LIKE 'L%'
            ) THEN 'active'
            ELSE 'archived'
        END,
        decision_reason = CASE
            WHEN EXISTS (
                SELECT 1 FROM logical_queries l
                WHERE l.search_domain_id = search_domains.search_domain_id
                  AND l.status = 'active'
                  AND l.logical_query_id LIKE 'L%'
            ) THEN decision_reason
            ELSE decision_reason || ' | no surviving logical query'
        END
        """
    )


def validate_search_frame(
    connection: sqlite3.Connection,
    fetcher: Any | None = None,
) -> Dict[str, Any]:
    """Validate PRESS, zero hits, and development/hidden seed recall."""
    _assert_not_frozen(connection)
    require_complete(connection, ["search_frame_derived"])
    hidden_search_status = hidden_seed_search_log_status(connection)
    if hidden_search_status["missing_routes"]:
        raise RuntimeError(
            "H2 hidden-seed provenance is incomplete; missing routes: "
            + ", ".join(hidden_search_status["missing_routes"])
        )
    validation_count = connection.execute(
        """
        SELECT COUNT(*) FROM evidence_seeds
        WHERE seed_role = 'validation'
          AND supplied_by = 'H2'
          AND hidden_during_development = 1
          AND eligibility_status = 'eligible'
          AND language = 'en'
        """
    ).fetchone()[0]
    if validation_count < 1:
        raise RuntimeError(
            "At least one eligible English H2 hidden validation seed is "
            "required"
        )
    logged_hidden_seed_dois: set[str] = set()
    for row in connection.execute(
        """
        SELECT eligible_seed_dois_json
        FROM hidden_seed_search_log
        WHERE reviewer_role = 'H2'
          AND completion_status = 'complete'
        """
    ):
        logged_hidden_seed_dois.update(
            normalize_doi(value)
            for value in json.loads(row["eligible_seed_dois_json"])
            if normalize_doi(value)
        )
    imported_hidden_seed_dois = {
        normalize_doi(row["doi"])
        for row in connection.execute(
            """
            SELECT doi FROM evidence_seeds
            WHERE seed_role = 'validation'
              AND supplied_by = 'H2'
              AND hidden_during_development = 1
              AND eligibility_status = 'eligible'
              AND language = 'en'
            """
        )
        if normalize_doi(row["doi"])
    }
    if logged_hidden_seed_dois != imported_hidden_seed_dois:
        logged_not_imported = sorted(
            logged_hidden_seed_dois - imported_hidden_seed_dois
        )
        imported_not_logged = sorted(
            imported_hidden_seed_dois - logged_hidden_seed_dois
        )
        raise RuntimeError(
            "H2 hidden-seed DOI reconciliation failed; "
            f"logged_not_imported={logged_not_imported}; "
            f"imported_not_logged={imported_not_logged}"
        )
    redundancy_archived = _resolve_redundancy_reviews(
        connection,
        fetcher=fetcher,
    )
    active_queries = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT physical_query_id FROM physical_queries
            WHERE status = 'active'
              AND logical_query_id LIKE 'L%'
            ORDER BY physical_query_id
            """
        )
    ]
    inventory_kwargs: Dict[str, Any] = {}
    if fetcher is not None:
        inventory_kwargs["fetcher"] = fetcher
    totals = inventory_physical_queries(
        connection,
        active_queries,
        "search_frame_validation_inventory",
        **inventory_kwargs,
    )
    archived_zero = _archive_zero_hit_families(connection, totals)
    active_logical = connection.execute(
        """
        SELECT logical_query_id, press_status FROM logical_queries
        WHERE status = 'active' AND logical_query_id LIKE 'L%'
        ORDER BY logical_query_id
        """
    ).fetchall()
    failed_press = [
        str(row["logical_query_id"])
        for row in active_logical
        if row["press_status"] != "pass"
    ]
    if failed_press:
        raise RuntimeError(
            "Active logical queries lack PRESS pass: "
            + ", ".join(failed_press)
        )
    physical = connection.execute(
        """
        SELECT physical_query_id, expression
        FROM physical_queries
        WHERE status = 'active' AND logical_query_id LIKE 'L%'
        ORDER BY physical_query_id
        """
    ).fetchall()
    keys = openalex_api_keys()
    check = fetcher or None
    seeds = connection.execute(
        """
        SELECT * FROM evidence_seeds
        ORDER BY seed_role, seed_id
        """
    ).fetchall()
    missed: List[str] = []
    recalled_count = 0
    provider_missing_count = 0
    supplemented_count = 0
    eligible_english: List[sqlite3.Row] = []
    for seed in seeds:
        if seed["eligibility_status"] != "eligible":
            connection.execute(
                """
                UPDATE evidence_seeds
                SET indexability_status = 'not_applicable',
                    recall_status = 'not_applicable'
                WHERE seed_id = ?
                """,
                (seed["seed_id"],),
            )
            continue
        if str(seed["language"]).casefold() != "en":
            connection.execute(
                """
                UPDATE evidence_seeds
                SET eligibility_status = 'excluded_non_english',
                    indexability_status = 'not_applicable',
                    recall_status = 'not_applicable',
                    nonrecall_reason = 'E_LANGUAGE_NON_ENGLISH'
                WHERE seed_id = ?
                """,
                (seed["seed_id"],),
            )
            continue
        if str(seed["recall_status"]) == "supplemented":
            supplemented_count += 1
            provider_missing_count += 1
            continue
        eligible_english.append(seed)

    request_count = 0
    seed_batch_size = int(
        read_json(SATURATION_PROTOCOL_PATH)["api"][
            "seed_recall_doi_batch_size"
        ]
    )
    if not 1 <= seed_batch_size <= 40:
        raise ValueError("seed_recall_doi_batch_size must be between 1 and 40")

    def batch_matches(
        doi_batch: Sequence[str],
        expression: str = "",
    ) -> set[str]:
        nonlocal request_count
        kwargs: Dict[str, Any] = {
            "api_key": (
                keys[request_count % len(keys)] if keys else ""
            )
        }
        if check is not None:
            kwargs["fetcher"] = check
        request_count += 1
        return batch_openalex_seed_matches(
            doi_batch,
            expression,
            **kwargs,
        )

    indexed_dois: set[str] = set()
    unique_dois = sorted(
        {
            normalize_doi(seed["doi"])
            for seed in eligible_english
            if normalize_doi(seed["doi"])
        }
    )
    for start in range(0, len(unique_dois), seed_batch_size):
        indexed_dois.update(
            batch_matches(unique_dois[start : start + seed_batch_size])
        )

    matches_by_doi: Dict[str, List[str]] = {
        doi: [] for doi in indexed_dois
    }
    indexed_list = sorted(indexed_dois)
    for query in physical:
        query_matches: set[str] = set()
        for start in range(0, len(indexed_list), seed_batch_size):
            query_matches.update(
                batch_matches(
                    indexed_list[start : start + seed_batch_size],
                    str(query["expression"]),
                )
            )
        for doi in sorted(query_matches):
            if doi in matches_by_doi:
                matches_by_doi[doi].append(
                    str(query["physical_query_id"])
                )

    for seed in eligible_english:
        doi = normalize_doi(seed["doi"])
        if doi not in indexed_dois:
            provider_missing_count += 1
            connection.execute(
                """
                UPDATE evidence_seeds
                SET indexability_status = 'provider_missing',
                    recall_status = 'supplement_required',
                    nonrecall_reason = CASE
                        WHEN nonrecall_reason = ''
                        THEN 'OPENALEX_NOT_INDEXABLE'
                        ELSE nonrecall_reason
                    END
                WHERE seed_id = ?
                """,
                (seed["seed_id"],),
            )
            continue
        matches = matches_by_doi.get(doi, [])
        status = "recalled" if matches else "missed_query_terms"
        if matches:
            recalled_count += 1
        else:
            missed.append(str(seed["seed_id"]))
        connection.execute(
            """
            UPDATE evidence_seeds
            SET indexability_status = 'indexable', recall_status = ?,
                recall_query_ids = ?,
                nonrecall_reason = CASE
                    WHEN ? = 'recalled' THEN ''
                    ELSE 'SEARCH_FRAME_TERM_GAP'
                END
            WHERE seed_id = ?
            """,
            (status, "|".join(matches), status, seed["seed_id"]),
        )
    connection.commit()
    active_k = connection.execute(
        """
        SELECT COUNT(DISTINCT search_domain_id)
        FROM logical_queries
        WHERE status = 'active' AND logical_query_id LIKE 'L%'
        """
    ).fetchone()[0]
    active_q = connection.execute(
        """
        SELECT COUNT(*) FROM logical_queries
        WHERE status = 'active' AND logical_query_id LIKE 'L%'
        """
    ).fetchone()[0]
    active_p = connection.execute(
        """
        SELECT COUNT(*) FROM physical_queries
        WHERE status = 'active' AND logical_query_id LIKE 'L%'
        """
    ).fetchone()[0]
    complete = not missed and active_q > 0 and active_p > 0
    details = {
        "K": active_k,
        "Q": active_q,
        "P": active_p,
        "validation_seeds": validation_count,
        "hidden_seed_search_provenance": hidden_search_status,
        "recalled_seeds": recalled_count,
        "provider_missing_seeds": provider_missing_count,
        "supplemented_provider_missing_seeds": supplemented_count,
        "seed_validation_api_requests": request_count,
        "missed_query_term_seeds": missed,
        "zero_hit_archived_queries": archived_zero,
        "result_set_redundancy_archived_queries": redundancy_archived,
    }
    set_stage(
        connection,
        "search_frame_validated",
        "complete" if complete else "blocked",
        details,
    )
    connection.commit()
    if missed:
        raise RuntimeError(
            "Indexable seeds were missed. Add only evidence-sourced English "
            "synonyms, re-code, version the frame, and validate again: "
            + ", ".join(missed)
        )
    return details


def freeze_search_frame(
    connection: sqlite3.Connection,
) -> Dict[str, Any]:
    """Freeze a deterministic, non-secret formal search-frame manifest."""
    require_complete(connection, ["search_frame_validated"])
    existing = connection.execute(
        "SELECT value FROM metadata WHERE key = 'search_frame_frozen_hash'"
    ).fetchone()
    if existing is not None and FORMAL_SEARCH_FRAME_PATH.exists():
        return read_json(FORMAL_SEARCH_FRAME_PATH)
    domains = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM search_domains ORDER BY search_domain_id"
        )
    ]
    logical = [
        dict(row)
        for row in connection.execute(
            """
            SELECT * FROM logical_queries
            WHERE logical_query_id LIKE 'L%'
            ORDER BY logical_query_id
            """
        )
    ]
    physical = [
        dict(row)
        for row in connection.execute(
            """
            SELECT * FROM physical_queries
            WHERE logical_query_id LIKE 'L%'
            ORDER BY physical_query_id
            """
        )
    ]
    seeds = [
        {
            key: row[key]
            for key in (
                "seed_id",
                "doi",
                "seed_role",
                "language",
                "eligibility_status",
                "indexability_status",
                "recall_status",
                "recall_query_ids",
                "nonrecall_reason",
            )
        }
        for row in connection.execute(
            "SELECT * FROM evidence_seeds ORDER BY seed_role, seed_id"
        )
    ]
    payload: Dict[str, Any] = {
        "schema_version": "3.4.0",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "domains": domains,
        "logical_queries": logical,
        "physical_queries": physical,
        "seed_recall": seeds,
        "counts": {
            "K": sum(row["status"] == "active" for row in domains),
            "Q": sum(row["status"] == "active" for row in logical),
            "P": sum(row["status"] == "active" for row in physical),
        },
    }
    payload["frozen_hash"] = json_hash(payload)
    write_json(FORMAL_SEARCH_FRAME_PATH, payload)
    connection.execute(
        """
        INSERT INTO metadata(key, value)
        VALUES ('search_frame_frozen_hash', ?)
        """,
        (payload["frozen_hash"],),
    )
    current_version = connection.execute(
        """
        SELECT frame_version FROM search_frame_versions
        WHERE status = 'current'
        """
    ).fetchone()
    if current_version is None:
        raise RuntimeError("No current search-frame version to freeze")
    version_body = {
        "frame_version": int(current_version["frame_version"]),
        "domains": domains,
        "logical_queries": logical,
        "physical_queries": physical,
        "seed_recall": seeds,
    }
    connection.execute(
        """
        UPDATE search_frame_versions
        SET frame_hash = ?, counts_json = ?, frame_json = ?,
            status = 'frozen'
        WHERE frame_version = ?
        """,
        (
            json_hash(version_body),
            json.dumps(payload["counts"], sort_keys=True),
            json.dumps(
                version_body,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            current_version["frame_version"],
        ),
    )
    _register_snapshot(
        connection,
        "frozen_search_frame_v3",
        FORMAL_SEARCH_FRAME_PATH,
        "frozen_search_frame",
    )
    set_stage(
        connection,
        "search_frame_frozen",
        "complete",
        {
            **payload["counts"],
            "frozen_hash": payload["frozen_hash"],
        },
    )
    connection.commit()
    return payload


def reopen_search_frame(
    connection: sqlite3.Connection,
    notes: str,
    reviewer_role: str = "H2",
) -> Dict[str, Any]:
    """Open a new frame version after adjudicated post-freeze evidence."""
    if reviewer_role.upper() != "H2":
        raise ValueError("Only H2 may authorize a new search-frame version")
    if not notes.strip():
        raise ValueError("Reopening a search frame requires H2 notes")
    frozen = connection.execute(
        "SELECT value FROM metadata WHERE key = 'search_frame_frozen_hash'"
    ).fetchone()
    if frozen is None or not FORMAL_SEARCH_FRAME_PATH.exists():
        raise RuntimeError("No frozen search frame is available to reopen")
    version = connection.execute(
        """
        SELECT frame_version FROM search_frame_versions
        WHERE status = 'frozen'
        ORDER BY frame_version DESC LIMIT 1
        """
    ).fetchone()
    if version is None:
        raise RuntimeError("Frozen search-frame version row is missing")
    frame_version = int(version["frame_version"])
    archive_dir = FORMAL_SEARCH_FRAME_PATH.parent / "search_frame_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_path = archive_dir / (
        f"frozen_search_frame_v{frame_version}_"
        f"{str(frozen['value'])[:12]}.json"
    )
    write_json(archived_path, read_json(FORMAL_SEARCH_FRAME_PATH))
    _register_snapshot(
        connection,
        f"frozen_search_frame_v{frame_version}",
        archived_path,
        "superseded_frozen_search_frame",
    )
    connection.execute(
        "DELETE FROM source_snapshots WHERE source_id = "
        "'frozen_search_frame_v3'"
    )
    connection.execute(
        "DELETE FROM metadata WHERE key = 'search_frame_frozen_hash'"
    )
    connection.execute(
        "DELETE FROM metadata WHERE key = 'formal_review_rank_offset'"
    )
    connection.execute(
        """
        UPDATE search_frame_versions
        SET status = 'superseded_by_postfreeze_evidence'
        WHERE frame_version = ?
        """,
        (frame_version,),
    )
    connection.execute(
        """
        UPDATE discovery_queries
        SET status = 'archived',
            archive_reason = 'superseded_formal_search_frame'
        WHERE query_role = 'formal_search_family'
          AND status = 'active'
        """
    )
    invalidate_stages(
        connection,
        (
            "terms_coded",
            "search_frame_derived",
            "search_frame_validated",
            "search_frame_frozen",
            "formal_retrieval_complete",
            "literature_screened",
            "indicators_extracted",
            "dimensions_derived",
            "features_selected",
            "audit_complete",
        ),
        "H2-authorized post-freeze term or indicator evidence",
    )
    log_event(
        connection,
        "search_frame_reopened",
        "search_frame_version",
        str(frame_version),
        {
            "reviewer_role": "H2",
            "notes": notes.strip(),
            "archived_path": str(archived_path),
        },
    )
    connection.commit()
    return {
        "superseded_frame_version": frame_version,
        "superseded_frozen_hash": str(frozen["value"]),
        "archived_path": str(archived_path),
        "notes": notes.strip(),
    }
