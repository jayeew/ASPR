from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import coding
import screening
from common import (
    ROOT,
    deterministic_ten_percent,
    iter_csv,
    json_hash,
    normalize_doi,
    normalize_term,
    or_block,
    parse_bool,
    read_json,
    sha256_bytes,
    term_match_key,
    utc_now,
    write_csv_iter,
)
from database import (
    assert_registered_review_attestation,
    connect,
    invalidate_stages,
    log_event,
    set_stage,
    snapshot_import_file,
)
from providers import (
    OPENALEX_BASE_URL,
    fetch_json,
    insert_openalex_record,
    openalex_api_keys,
    openalex_record,
    query_definition_hash,
    retrieve_physical_query,
    safe_provider_error,
)


SATURATION_PROTOCOL_PATH = ROOT / "saturation_protocol_v4.json"
DISCOVERY_STOP_AMENDMENT_PATH = (
    ROOT / "protocol_amendment_round15_pragmatic_stop_v4.json"
)
BOOTSTRAP_PATH = ROOT / "bootstrap_query_v4.json"
PROTOCOL_PATH = ROOT / "protocol_v4.json"
DISCOVERY_EXTRACTION_FIELDS = (
    "record_key",
    "doi",
    "title",
    "abstract",
    "review_round",
    "item_type",
    "verbatim_name",
    "location",
    "evidence_span",
    "proposed_role",
    "status",
    "exclusion_reason",
    "extractor_role",
    "canonical_family_label",
    "h1_decision",
    "h2_decision",
    "record_extraction_complete",
    "no_relevant_items",
    "review_notes",
)
DISCOVERY_INDICATOR_ADJUDICATION_FIELDS = (
    "candidate_id",
    "record_key",
    "review_round",
    "doi",
    "title",
    "abstract",
    "raw_name_en",
    "location",
    "evidence_span",
    "proposed_role",
    "extracted_by",
    "h1_decision",
    "h2_decision",
    "canonical_family_label",
    "adjudication_notes",
)
DISCOVERY_PROPOSED_ROLES = {
    "construct",
    "indicator_or_measure",
    "t0_predictor",
    "opportunity_or_context",
    "control",
    "validation_outcome",
}


class DailyBudgetExhausted(RuntimeError):
    """Raised when every configured OpenAlex key has exhausted daily credit."""


def _stable_seed(namespace: str, label: str) -> int:
    digest = sha256_bytes(f"{namespace}|{label}".encode("utf-8"))
    return int(digest[:8], 16) % 2_147_483_647


def _slug(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() else "_"
        for character in value.upper()
    )
    return "_".join(part for part in cleaned.split("_") if part)


def _date_filters(
    from_year: int | None,
    to_year: int | None,
    cutoff_date: str,
) -> List[str]:
    filters: List[str] = []
    if from_year is not None:
        filters.append(f"from_publication_date:{from_year:04d}-01-01")
    upper = cutoff_date
    if to_year is not None and to_year < int(cutoff_date[:4]):
        upper = f"{to_year:04d}-12-31"
    filters.append(f"to_publication_date:{upper}")
    return filters


def _query_rows() -> List[Dict[str, Any]]:
    saturation_protocol = read_json(SATURATION_PROTOCOL_PATH)
    bootstrap = read_json(BOOTSTRAP_PATH)
    protocol = read_json(PROTOCOL_PATH)
    sampling = saturation_protocol["sampling"]
    namespace = str(sampling["random_seed_namespace"])
    object_block = or_block(bootstrap["object_terms"])
    target_block = or_block(bootstrap["target_terms"])
    evidence_block = or_block(bootstrap["evidence_terms"])
    complete_expression = " AND ".join(
        (object_block, target_block, evidence_block)
    )
    rows: List[Dict[str, Any]] = []

    def append_query(
        query_id: str,
        role: str,
        label: str,
        expression: str,
        filters: Sequence[str],
        sample_size: int,
    ) -> None:
        seed = _stable_seed(namespace, query_id)
        body = {
            "expression": expression,
            "filter_expression": ",".join(filters),
            "sample_size": sample_size,
            "random_seed": seed,
        }
        rows.append(
            {
                "discovery_query_id": query_id,
                "query_role": role,
                "stratum_label": label,
                **body,
                "query_hash": json_hash(body),
                "source_ids": [],
                "source_dois": [],
                "source_phrases": [],
                "derivation_rule": "frozen_domain_free_sampling_block",
            }
        )

    cutoff_date = str(protocol["cutoff_date"])
    for band in sampling["publication_year_bands"]:
        for work_type in sampling["work_types"]:
            label = f"{band['label']}|{work_type}"
            filters = _date_filters(
                band.get("from_year"),
                band.get("to_year"),
                cutoff_date,
            )
            filters.append(f"type:{work_type}")
            append_query(
                f"DS_BASE_{_slug(str(band['label']))}_{_slug(work_type)}",
                "base_year_type",
                label,
                complete_expression,
                filters,
                int(sampling["base_sample_per_stratum"]),
            )
    common_filters = [f"to_publication_date:{cutoff_date}"]
    for term in bootstrap["target_terms"]:
        for work_type in sampling["work_types"]:
            expression = " AND ".join(
                (object_block, or_block([term]), evidence_block)
            )
            append_query(
                f"DS_TARGET_{_slug(term)}_{_slug(work_type)}",
                "target_oversample",
                f"{term}|{work_type}",
                expression,
                [*common_filters, f"type:{work_type}"],
                int(sampling["target_oversample_per_stratum"]),
            )
    for term in bootstrap["evidence_terms"]:
        for work_type in sampling["work_types"]:
            expression = " AND ".join(
                (object_block, target_block, or_block([term]))
            )
            append_query(
                f"DS_EVIDENCE_{_slug(term)}_{_slug(work_type)}",
                "evidence_oversample",
                f"{term}|{work_type}",
                expression,
                [*common_filters, f"type:{work_type}"],
                int(sampling["evidence_oversample_per_stratum"]),
            )
    evidence_source = str(
        sampling.get("development_evidence_formula_source") or ""
    ).strip()
    if not evidence_source:
        return rows
    evidence_path = (ROOT / evidence_source).resolve()
    evidence = read_json(evidence_path)
    phrase_sources: Dict[str, Dict[str, Any]] = {}
    for source in evidence.get("records", []):
        if not isinstance(source, dict):
            continue
        for raw_phrase in source.get("formula_authorization", []):
            phrase = str(raw_phrase or "").strip()
            if not phrase:
                continue
            key = normalize_term(phrase)
            item = phrase_sources.setdefault(
                key,
                {
                    "phrase": phrase,
                    "source_ids": set(),
                    "source_dois": set(),
                },
            )
            item["source_ids"].add(str(source.get("source_id") or ""))
            item["source_dois"].add(str(source.get("doi") or ""))
    for key in sorted(phrase_sources):
        item = phrase_sources[key]
        phrase_hash = sha256_bytes(key.encode("utf-8"))[:8].upper()
        query_id = (
            f"DS_DEVFORMULA_{_slug(key)[:45]}_{phrase_hash}"
        )
        expression = " AND ".join(
            (object_block, target_block, or_block([item["phrase"]]))
        )
        append_query(
            query_id,
            "development_formula_oversample",
            str(item["phrase"]),
            expression,
            [f"to_publication_date:{cutoff_date}", "type:article|review"],
            int(
                sampling[
                    "development_evidence_formula_oversample_per_stratum"
                ]
            ),
        )
        rows[-1]["source_ids"] = sorted(
            value for value in item["source_ids"] if value
        )
        rows[-1]["source_dois"] = sorted(
            value for value in item["source_dois"] if value
        )
        rows[-1]["source_phrases"] = [str(item["phrase"])]
        rows[-1]["derivation_rule"] = (
            "deduplicated formula_authorization phrase from frozen "
            "53-paper development evidence"
        )
    return rows


def derive_discovery_queries(connection: sqlite3.Connection) -> Dict[str, Any]:
    """Create domain-free deterministic sampling strata from frozen blocks."""
    rows = _query_rows()
    active_ids = [str(row["discovery_query_id"]) for row in rows]
    for row in rows:
        database_row = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "source_ids",
                "source_dois",
                "source_phrases",
                "derivation_rule",
            }
        }
        connection.execute(
            """
            INSERT INTO discovery_queries(
                discovery_query_id, query_role, stratum_label, expression,
                filter_expression, sample_size, random_seed, query_hash,
                status, archive_reason
            ) VALUES (
                :discovery_query_id, :query_role, :stratum_label,
                :expression, :filter_expression, :sample_size,
                :random_seed, :query_hash, 'active', ''
            )
            ON CONFLICT(discovery_query_id) DO UPDATE SET
                query_role = excluded.query_role,
                stratum_label = excluded.stratum_label,
                expression = excluded.expression,
                filter_expression = excluded.filter_expression,
                sample_size = excluded.sample_size,
                random_seed = excluded.random_seed,
                query_hash = excluded.query_hash,
                status = 'active',
                archive_reason = ''
            """,
            database_row,
        )
        if row["source_ids"]:
            connection.execute(
                """
                INSERT INTO discovery_query_evidence(
                    discovery_query_id, source_ids_json,
                    source_dois_json, source_phrases_json,
                    derivation_rule
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(discovery_query_id) DO UPDATE SET
                    source_ids_json = excluded.source_ids_json,
                    source_dois_json = excluded.source_dois_json,
                    source_phrases_json = excluded.source_phrases_json,
                    derivation_rule = excluded.derivation_rule
                """,
                (
                    row["discovery_query_id"],
                    json.dumps(row["source_ids"], ensure_ascii=False),
                    json.dumps(row["source_dois"], ensure_ascii=False),
                    json.dumps(row["source_phrases"], ensure_ascii=False),
                    row["derivation_rule"],
                ),
            )
    placeholders = ",".join("?" for _ in active_ids)
    connection.execute(
        f"""
        UPDATE discovery_queries
        SET status = 'archived', archive_reason = 'not_in_frozen_v3_1_design'
        WHERE discovery_query_id NOT IN ({placeholders})
          AND query_role NOT IN (
              'development_citation_network',
              'citation_tracking_network',
              'formal_search_family'
          )
        """,
        active_ids,
    )
    expected_results = sum(int(row["sample_size"]) for row in rows)
    details = {
        "discovery_strata": len(rows),
        "base_strata": sum(row["query_role"] == "base_year_type" for row in rows),
        "target_oversample_strata": sum(
            row["query_role"] == "target_oversample" for row in rows
        ),
        "evidence_oversample_strata": sum(
            row["query_role"] == "evidence_oversample" for row in rows
        ),
        "development_formula_oversample_strata": sum(
            row["query_role"] == "development_formula_oversample"
            for row in rows
        ),
        "maximum_rows_before_deduplication": expected_results,
        "design_hash": json_hash({"queries": rows}),
    }
    log_event(
        connection,
        "discovery_frame_derived",
        "collection",
        "domain_free_saturation_strata",
        details,
    )
    connection.commit()
    return details


def _insert_bootstrap_logical_query(
    connection: sqlite3.Connection,
    logical_query_id: str,
    family_label: str,
    filter_expression: str,
) -> None:
    query_hash = query_definition_hash("", filter_expression)
    connection.execute(
        """
        INSERT INTO logical_queries(
            logical_query_id, query_version, search_domain_id,
            family_label, logical_expression, object_terms_json,
            domain_terms_json, context_terms_json, status,
            archive_reason, press_status, query_hash
        ) VALUES (?, 1, 'BOOTSTRAP_CITATION', ?, '', '[]', '[]', '[]',
                  'bootstrap', '', 'not_applicable', ?)
        ON CONFLICT(logical_query_id) DO UPDATE SET
            family_label = excluded.family_label,
            logical_expression = '',
            status = 'bootstrap',
            archive_reason = '',
            query_hash = excluded.query_hash
        """,
        (logical_query_id, family_label, query_hash),
    )


def _insert_bootstrap_physical_query(
    connection: sqlite3.Connection,
    physical_query_id: str,
    logical_query_id: str,
    filter_expression: str,
) -> None:
    connection.execute(
        """
        INSERT INTO physical_queries(
            physical_query_id, logical_query_id, provider, expression,
            filter_expression, status, query_hash
        ) VALUES (?, ?, 'OpenAlex', '', ?, 'active', ?)
        ON CONFLICT(physical_query_id) DO UPDATE SET
            filter_expression = excluded.filter_expression,
            status = 'active',
            query_hash = excluded.query_hash
        """,
        (
            physical_query_id,
            logical_query_id,
            filter_expression,
            query_definition_hash("", filter_expression),
        ),
    )


def _development_seed_records(
    connection: sqlite3.Connection,
) -> List[sqlite3.Row]:
    return connection.execute(
        """
        SELECT e.seed_id, e.doi AS seed_doi, r.*
        FROM evidence_seeds e
        JOIN records r ON r.doi = e.doi
        WHERE e.seed_role = 'development'
          AND e.eligibility_status = 'eligible'
        ORDER BY e.seed_id
        """
    ).fetchall()


def register_development_citation_queries(
    connection: sqlite3.Connection,
) -> Dict[str, Any]:
    """Register complete forward/backward seed-network requests."""
    protocol = read_json(PROTOCOL_PATH)
    sources = _development_seed_records(connection)
    if len(sources) != connection.execute(
        """
        SELECT COUNT(*) FROM evidence_seeds
        WHERE seed_role = 'development'
          AND eligibility_status = 'eligible'
        """
    ).fetchone()[0]:
        raise RuntimeError("Development seeds must be hydrated first")
    source_ids = sorted(
        {
            str(row["provider_id"]).rsplit("/", maxsplit=1)[-1]
            for row in sources
        }
    )
    source_dois = sorted(str(row["seed_doi"]) for row in sources)
    references = sorted(
        {
            str(reference).rsplit("/", maxsplit=1)[-1]
            for row in sources
            for reference in json.loads(row["referenced_works_json"])
        }
    )
    cutoff = str(protocol["cutoff_date"])
    forward_logical = "B0002_DEVELOPMENT_SEED_FORWARD_CITATIONS"
    forward_filter = (
        "cites:"
        + "|".join(source_ids)
        + f",to_publication_date:{cutoff},type:article|review"
    )
    _insert_bootstrap_logical_query(
        connection,
        forward_logical,
        "development_seed_forward_citations",
        forward_filter,
    )
    forward_physical = f"{forward_logical}__P001"
    _insert_bootstrap_physical_query(
        connection,
        forward_physical,
        forward_logical,
        forward_filter,
    )
    backward_logical = "B0003_DEVELOPMENT_SEED_BACKWARD_REFERENCES"
    _insert_bootstrap_logical_query(
        connection,
        backward_logical,
        "development_seed_backward_references",
        "ids.openalex:[chunk],type:article|review",
    )
    backward_physical: List[str] = []
    for index, start in enumerate(range(0, len(references), 100), start=1):
        chunk = references[start : start + 100]
        physical_id = f"{backward_logical}__P{index:03d}"
        filter_expression = (
            "ids.openalex:"
            + "|".join(chunk)
            + ",type:article|review"
        )
        _insert_bootstrap_physical_query(
            connection,
            physical_id,
            backward_logical,
            filter_expression,
        )
        backward_physical.append(physical_id)
    if backward_physical:
        placeholders = ",".join("?" for _ in backward_physical)
        connection.execute(
            f"""
            UPDATE physical_queries
            SET status = 'archived'
            WHERE logical_query_id = ?
              AND physical_query_id NOT IN ({placeholders})
            """,
            [backward_logical, *backward_physical],
        )
    details = {
        "seed_sources": len(sources),
        "unique_seed_ids": len(source_ids),
        "unique_backward_reference_ids": len(references),
        "forward_physical_queries": [forward_physical],
        "backward_physical_queries": backward_physical,
        "source_dois": source_dois,
    }
    connection.commit()
    return details


def _register_citation_discovery_query(
    connection: sqlite3.Connection,
    query_id: str,
    label: str,
    source_ids: Sequence[str],
    source_dois: Sequence[str],
) -> None:
    body = {
        "query_id": query_id,
        "label": label,
        "source_ids": list(source_ids),
        "source_dois": list(source_dois),
    }
    connection.execute(
        """
        INSERT INTO discovery_queries(
            discovery_query_id, query_role, stratum_label, expression,
            filter_expression, sample_size, random_seed, query_hash,
            status, archive_reason
        ) VALUES (?, 'development_citation_network', ?, '', '', 0, 0, ?,
                  'network', '')
        ON CONFLICT(discovery_query_id) DO UPDATE SET
            stratum_label = excluded.stratum_label,
            query_hash = excluded.query_hash,
            status = 'network',
            archive_reason = ''
        """,
        (query_id, label, json_hash(body)),
    )
    connection.execute(
        """
        INSERT INTO discovery_query_evidence(
            discovery_query_id, source_ids_json, source_dois_json,
            source_phrases_json, derivation_rule
        ) VALUES (?, ?, ?, '[]', ?)
        ON CONFLICT(discovery_query_id) DO UPDATE SET
            source_ids_json = excluded.source_ids_json,
            source_dois_json = excluded.source_dois_json,
            derivation_rule = excluded.derivation_rule
        """,
        (
            query_id,
            json.dumps(list(source_ids), ensure_ascii=False),
            json.dumps(list(source_dois), ensure_ascii=False),
            "complete citation network of frozen development evidence",
        ),
    )


def _map_citation_discovery_hits(
    connection: sqlite3.Connection,
    query_id: str,
    physical_query_ids: Sequence[str],
) -> int:
    if not physical_query_ids:
        return 0
    placeholders = ",".join("?" for _ in physical_query_ids)
    rows = connection.execute(
        f"""
        SELECT DISTINCT record_key
        FROM query_hits
        WHERE run_role = 'bootstrap_citation'
          AND physical_query_id IN ({placeholders})
        ORDER BY record_key
        """,
        list(physical_query_ids),
    ).fetchall()
    for rank, row in enumerate(rows, start=1):
        record_key = str(row["record_key"])
        connection.execute(
            """
            INSERT OR IGNORE INTO discovery_hits(
                discovery_query_id, record_key, sample_rank,
                selection_hash, review_rank, review_round, review_status
            ) VALUES (?, ?, ?, ?, 0, 0, 'unassigned')
            """,
            (
                query_id,
                record_key,
                rank,
                _selection_hash(query_id, record_key),
            ),
        )
    connection.execute(
        """
        UPDATE discovery_queries
        SET sample_size = ?
        WHERE discovery_query_id = ?
        """,
        (len(rows), query_id),
    )
    return len(rows)


def _persist_seed_citation_edges(
    connection: sqlite3.Connection,
    sources: Sequence[sqlite3.Row],
) -> Dict[str, int]:
    source_by_id = {
        str(row["provider_id"]): str(row["record_key"])
        for row in sources
    }
    backward = 0
    for source in sources:
        for target in json.loads(source["referenced_works_json"]):
            connection.execute(
                """
                INSERT OR IGNORE INTO citation_edges(
                    source_record_key, target_provider_id, direction,
                    iteration, eligibility_status
                ) VALUES (?, ?, 'backward', 0, 'pending_screening')
                """,
                (source["record_key"], target),
            )
            backward += 1
    forward = 0
    for target in connection.execute(
        """
        SELECT DISTINCT r.provider_id, r.referenced_works_json
        FROM records r
        JOIN query_hits h ON h.record_key = r.record_key
        WHERE h.run_role = 'bootstrap_citation'
          AND h.physical_query_id LIKE
              'B0002_DEVELOPMENT_SEED_FORWARD_CITATIONS%'
        """
    ):
        references = set(json.loads(target["referenced_works_json"]))
        for source_id in sorted(references.intersection(source_by_id)):
            connection.execute(
                """
                INSERT OR IGNORE INTO citation_edges(
                    source_record_key, target_provider_id, direction,
                    iteration, eligibility_status
                ) VALUES (?, ?, 'forward', 0, 'pending_screening')
                """,
                (source_by_id[source_id], target["provider_id"]),
            )
            forward += 1
    return {"backward_edges": backward, "forward_edges": forward}


def expand_development_seed_citations(
    connection: sqlite3.Connection,
) -> Dict[str, Any]:
    """Retrieve the complete citation neighborhood of development seeds."""
    definitions = register_development_citation_queries(connection)
    physical_ids = [
        *definitions["forward_physical_queries"],
        *definitions["backward_physical_queries"],
    ]
    runs = [
        retrieve_physical_query(
            connection,
            physical_id,
            "bootstrap_citation",
        )
        for physical_id in physical_ids
    ]
    source_ids = [
        str(row["provider_id"])
        for row in _development_seed_records(connection)
    ]
    source_dois = list(definitions["source_dois"])
    forward_discovery = "DS_DEVELOPMENT_SEED_FORWARD_NETWORK"
    backward_discovery = "DS_DEVELOPMENT_SEED_BACKWARD_NETWORK"
    _register_citation_discovery_query(
        connection,
        forward_discovery,
        "development seed forward citations",
        source_ids,
        source_dois,
    )
    _register_citation_discovery_query(
        connection,
        backward_discovery,
        "development seed backward references",
        source_ids,
        source_dois,
    )
    forward_hits = _map_citation_discovery_hits(
        connection,
        forward_discovery,
        definitions["forward_physical_queries"],
    )
    backward_hits = _map_citation_discovery_hits(
        connection,
        backward_discovery,
        definitions["backward_physical_queries"],
    )
    edge_counts = _persist_seed_citation_edges(
        connection,
        _development_seed_records(connection),
    )
    details = {
        "physical_queries": len(physical_ids),
        "complete_physical_queries": sum(
            int(run["complete"]) for run in runs
        ),
        "forward_records": forward_hits,
        "backward_records": backward_hits,
        **edge_counts,
    }
    log_event(
        connection,
        "development_seed_citation_expansion",
        "collection",
        "development_seeds",
        details,
    )
    connection.commit()
    return details


def hydrate_development_seeds(
    connection: sqlite3.Connection,
) -> Dict[str, Any]:
    """Retrieve seed metadata in batched, inexpensive DOI-filter requests."""
    seeds = connection.execute(
        """
        SELECT * FROM evidence_seeds
        WHERE seed_role = 'development'
          AND eligibility_status = 'eligible'
        ORDER BY seed_id
        """
    ).fetchall()
    keys = openalex_api_keys()
    if not keys:
        raise RuntimeError("No OpenAlex API key is configured")
    found_dois: set[str] = set()
    requests = 0
    for start in range(0, len(seeds), 40):
        chunk = seeds[start : start + 40]
        dois = [normalize_doi(seed["doi"]) for seed in chunk]
        dois = [doi for doi in dois if doi]
        parameters = {
            "filter": "doi:" + "|".join(dois),
            "per_page": 100,
            "select": (
                "id,doi,display_name,publication_year,type,language,"
                "abstract_inverted_index,primary_location,best_oa_location,"
                "referenced_works"
            ),
            "api_key": keys[requests % len(keys)],
        }
        url = (
            f"{OPENALEX_BASE_URL}/works?"
            + urllib.parse.urlencode(parameters)
        )
        payload = fetch_json(url, retries=3, timeout_seconds=60)
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError("Malformed OpenAlex seed batch response")
        for item in results:
            if not isinstance(item, dict):
                continue
            record = openalex_record(
                item,
                retrieval_route="development_seed_hydration",
            )
            insert_openalex_record(connection, record)
            if record["doi"]:
                found_dois.add(str(record["doi"]))
        requests += 1
        connection.commit()
    snapshot_config = read_json(SATURATION_PROTOCOL_PATH)["snapshot"]
    index_path = Path(str(snapshot_config.get("works_id_index") or ""))
    snapshot_ids: set[str] = set()
    if index_path.is_file():
        provider_ids = [
            str(row[0]).rsplit("/", maxsplit=1)[-1]
            for row in connection.execute(
                """
                SELECT DISTINCT r.provider_id
                FROM evidence_seeds e
                JOIN records r ON r.doi = e.doi
                WHERE e.seed_role = 'development'
                """
            )
        ]
        local_index = sqlite3.connect(
            f"file:{index_path.resolve()}?mode=ro",
            uri=True,
            timeout=60,
        )
        try:
            for start in range(0, len(provider_ids), 500):
                chunk = provider_ids[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                snapshot_ids.update(
                    str(row[0])
                    for row in local_index.execute(
                        f"""
                        SELECT short_id FROM works_index
                        WHERE short_id IN ({placeholders})
                        """,
                        chunk,
                    )
                )
        finally:
            local_index.close()
    for seed in seeds:
        doi = normalize_doi(seed["doi"])
        provider_row = connection.execute(
            "SELECT provider_id FROM records WHERE doi = ? LIMIT 1",
            (doi,),
        ).fetchone()
        short_id = (
            str(provider_row[0]).rsplit("/", maxsplit=1)[-1]
            if provider_row is not None
            else ""
        )
        if doi in found_dois and short_id in snapshot_ids:
            indexability = "snapshot_and_api"
        elif doi in found_dois:
            indexability = "api_only"
        else:
            indexability = "not_indexed"
        connection.execute(
            """
            UPDATE evidence_seeds
            SET indexability_status = ?,
                nonrecall_reason = CASE
                    WHEN ? != 'not_indexed' THEN ''
                    ELSE 'OPENALEX_DOI_NOT_RETURNED_IN_BATCH_LOOKUP'
                END
            WHERE seed_id = ?
            """,
            (
                indexability,
                indexability,
                seed["seed_id"],
            ),
        )
    details = {
        "development_seeds": len(seeds),
        "found_by_doi": len(found_dois),
        "not_found": len(seeds) - len(found_dois),
        "found_in_local_snapshot_index": len(snapshot_ids),
        "local_snapshot_index": str(index_path) if index_path.is_file() else "",
        "batch_requests": requests,
    }
    log_event(
        connection,
        "development_seed_hydration",
        "collection",
        "development_seeds",
        details,
    )
    connection.commit()
    return details


def _rate_limit_url(api_key: str) -> str:
    return (
        f"{OPENALEX_BASE_URL}/rate-limit?"
        + urllib.parse.urlencode({"api_key": api_key})
    )


class OpenAlexBudgetScheduler:
    """Round-robin API keys while respecting reported daily credits."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.keys = openalex_api_keys()
        if not self.keys:
            raise RuntimeError("No OpenAlex API key is configured")
        self.remaining: Dict[int, float] = {}
        self.resets_at: Dict[int, str] = {}
        self.next_slot = 0
        self.lock = threading.Lock()
        self.refresh()

    def refresh(self) -> None:
        """Refresh and persist non-secret budget observations."""
        available_slots = 0
        for slot, key in enumerate(self.keys):
            try:
                payload = fetch_json(
                    _rate_limit_url(key),
                    retries=2,
                    timeout_seconds=30,
                )
            except Exception as error:
                self.remaining[slot] = 0.0
                self.resets_at[slot] = ""
                log_event(
                    self.connection,
                    "openalex_key_slot_unavailable",
                    "key_slot",
                    str(slot + 1),
                    {"error": safe_provider_error(error)},
                )
                continue
            rate = payload.get("rate_limit")
            if not isinstance(rate, dict):
                self.remaining[slot] = 0.0
                self.resets_at[slot] = ""
                log_event(
                    self.connection,
                    "openalex_key_slot_unavailable",
                    "key_slot",
                    str(slot + 1),
                    {"error": "malformed_rate_limit_response"},
                )
                continue
            remaining = float(rate.get("daily_remaining_usd") or 0.0)
            available_slots += int(remaining > 0)
            self.remaining[slot] = remaining
            self.resets_at[slot] = str(rate.get("resets_at") or "")
            self.connection.execute(
                """
                INSERT INTO api_budget_observations(
                    key_slot, daily_budget_usd, daily_used_usd,
                    daily_remaining_usd, resets_at, observed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    slot + 1,
                    float(rate.get("daily_budget_usd") or 0.0),
                    float(rate.get("daily_used_usd") or 0.0),
                    remaining,
                    self.resets_at[slot],
                    utc_now(),
                ),
            )
        self.connection.commit()
        if available_slots == 0:
            if os.getenv("OPENALEX_ALLOW_ANONYMOUS_FALLBACK") == "1":
                self.keys = [""]
                self.remaining = {0: float("inf")}
                self.resets_at = {0: ""}
                log_event(
                    self.connection,
                    "openalex_anonymous_fallback_enabled",
                    "provider_access",
                    "OpenAlex",
                    {
                        "reason": "configured_key_slots_reported_no_credit",
                        "key_values_persisted": False,
                    },
                )
                self.connection.commit()
                return
            raise DailyBudgetExhausted(
                "No configured OpenAlex key slot has reported daily credit"
            )

    def _available_slot(
        self,
        minimum_cost: float,
        excluded_slots: set[int] | None = None,
    ) -> int:
        excluded = excluded_slots or set()
        for offset in range(len(self.keys)):
            slot = (self.next_slot + offset) % len(self.keys)
            if (
                slot not in excluded
                and self.remaining.get(slot, 0.0) + 1e-12 >= minimum_cost
            ):
                self.next_slot = (slot + 1) % len(self.keys)
                return slot
        reset = min(
            (value for value in self.resets_at.values() if value),
            default="next UTC reset",
        )
        raise DailyBudgetExhausted(
            f"OpenAlex daily free credits exhausted; resume after {reset}"
        )

    def fetch_search(self, parameters: Mapping[str, Any]) -> Dict[str, Any]:
        """Execute a search request without exposing its selected key."""
        attempted: set[int] = set()
        last_error: Exception | None = None
        while len(attempted) < len(self.keys):
            with self.lock:
                slot = self._available_slot(0.001, attempted)
                self.remaining[slot] = max(
                    0.0,
                    self.remaining.get(slot, 0.0) - 0.001,
                )
            attempted.add(slot)
            request_parameters = dict(parameters)
            if self.keys[slot]:
                request_parameters["api_key"] = self.keys[slot]
            url = (
                f"{OPENALEX_BASE_URL}/works?"
                + urllib.parse.urlencode(request_parameters)
            )
            try:
                payload = fetch_json(
                    url,
                    retries=2,
                    timeout_seconds=60,
                )
            except Exception as error:
                last_error = error
                continue
            meta = payload.get("meta")
            cost = 0.001
            if isinstance(meta, dict) and isinstance(
                meta.get("cost_usd"),
                (int, float),
            ):
                cost = float(meta["cost_usd"])
            with self.lock:
                self.remaining[slot] = max(
                    0.0,
                    self.remaining.get(slot, 0.0) + 0.001 - cost,
                )
            return payload
        raise RuntimeError(
            "Every configured OpenAlex key failed the request: "
            + safe_provider_error(last_error or RuntimeError("unknown"))
        )


def _sample_parameters(
    row: sqlite3.Row,
    page: int,
    per_page: int,
) -> Dict[str, Any]:
    return {
        "search": str(row["expression"]),
        "filter": str(row["filter_expression"]),
        "sample": int(row["sample_size"]),
        "seed": int(row["random_seed"]),
        "page": page,
        "per_page": per_page,
        "select": (
            "id,doi,display_name,publication_year,type,language,"
            "abstract_inverted_index,primary_location,best_oa_location,"
            "referenced_works"
        ),
    }


def _selection_hash(query_id: str, record_key: str) -> str:
    protocol = read_json(SATURATION_PROTOCOL_PATH)
    namespace = protocol["sampling"]["random_seed_namespace"]
    return sha256_bytes(
        f"{namespace}|{query_id}|{record_key}".encode("utf-8")
    )


def _save_discovery_page(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    page: int,
    results: Sequence[Any],
) -> int:
    page_start = (page - 1) * 100
    inserted = 0
    bootstrap_physical_id = "B0001_DOMAIN_AGNOSTIC_BOOTSTRAP__P001"
    for offset, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        route_prefix = (
            "formal_saturation"
            if row["query_role"] == "formal_search_family"
            else "bootstrap_saturation"
        )
        record = openalex_record(
            item,
            retrieval_route=(
                f"{route_prefix}:{row['discovery_query_id']}"
            ),
        )
        insert_openalex_record(connection, record)
        rank = page_start + offset
        review_rank = 0
        if row["query_role"] == "formal_search_family":
            rank_offset = connection.execute(
                """
                SELECT value FROM metadata
                WHERE key = 'formal_review_rank_offset'
                """
            ).fetchone()
            review_rank = rank + (
                int(rank_offset["value"]) if rank_offset is not None else 0
            )
        connection.execute(
            """
            INSERT OR IGNORE INTO discovery_hits(
                discovery_query_id, record_key, sample_rank,
                selection_hash, review_rank, review_round, review_status
            ) VALUES (?, ?, ?, ?, ?, 0, 'unassigned')
            """,
            (
                row["discovery_query_id"],
                record["record_key"],
                rank,
                _selection_hash(
                    str(row["discovery_query_id"]),
                    str(record["record_key"]),
                ),
                review_rank,
            ),
        )
        if row["query_role"] != "formal_search_family":
            connection.execute(
                """
                INSERT OR IGNORE INTO query_hits(
                    provider, physical_query_id, run_role, record_key, rank
                ) VALUES ('OpenAlex', ?, 'bootstrap', ?, ?)
                """,
                (bootstrap_physical_id, record["record_key"], rank),
            )
        inserted += 1
    return inserted


def _save_discovery_checkpoint(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    sample_total: int,
    retrieved_rows: int,
    pages: int,
    next_page: int,
    complete: bool,
    stopped_reason: str = "",
    error: str = "",
) -> None:
    unique_hits = connection.execute(
        """
        SELECT COUNT(*) FROM discovery_hits
        WHERE discovery_query_id = ?
        """,
        (row["discovery_query_id"],),
    ).fetchone()[0]
    connection.execute(
        """
        INSERT INTO discovery_query_runs(
            discovery_query_id, query_hash, reported_sample_total,
            retrieved_rows, unique_hits, pages, next_page, complete,
            stopped_reason, error, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(discovery_query_id) DO UPDATE SET
            query_hash = excluded.query_hash,
            reported_sample_total = excluded.reported_sample_total,
            retrieved_rows = excluded.retrieved_rows,
            unique_hits = excluded.unique_hits,
            pages = excluded.pages,
            next_page = excluded.next_page,
            complete = excluded.complete,
            stopped_reason = excluded.stopped_reason,
            error = excluded.error,
            updated_at = excluded.updated_at
        """,
        (
            row["discovery_query_id"],
            row["query_hash"],
            sample_total,
            retrieved_rows,
            unique_hits,
            pages,
            next_page,
            int(complete),
            stopped_reason,
            error,
            utc_now(),
        ),
    )
    connection.commit()


def _retrieve_one_discovery_query(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    scheduler: OpenAlexBudgetScheduler,
    per_page: int,
    concurrency: int,
    maximum_page: int | None = None,
) -> Dict[str, Any]:
    run = connection.execute(
        """
        SELECT * FROM discovery_query_runs
        WHERE discovery_query_id = ?
        """,
        (row["discovery_query_id"],),
    ).fetchone()
    if run is not None and bool(run["complete"]):
        return dict(run)
    if run is not None and run["query_hash"] != row["query_hash"]:
        raise RuntimeError(
            "Discovery query changed during resumable retrieval: "
            f"{row['discovery_query_id']}"
        )
    page = int(run["next_page"]) if run is not None else 1
    retrieved_rows = int(run["retrieved_rows"]) if run is not None else 0
    pages = int(run["pages"]) if run is not None else 0
    sample_total = (
        int(run["reported_sample_total"])
        if run is not None and run["reported_sample_total"] is not None
        else -1
    )
    while True:
        if maximum_page is not None and page > maximum_page:
            _save_discovery_checkpoint(
                connection,
                row,
                max(sample_total, 0),
                retrieved_rows,
                pages,
                page,
                False,
                "review_capacity_loaded",
            )
            break
        if sample_total >= 0:
            expected_pages = math.ceil(sample_total / per_page)
            batch_pages = list(
                range(page, min(page + concurrency, expected_pages + 1))
            )
        else:
            batch_pages = [page]
        if maximum_page is not None:
            batch_pages = [
                value for value in batch_pages if value <= maximum_page
            ]
        if not batch_pages:
            _save_discovery_checkpoint(
                connection,
                row,
                max(sample_total, 0),
                retrieved_rows,
                pages,
                page,
                False,
                "review_capacity_loaded",
            )
            break
        try:
            payloads: Dict[int, Dict[str, Any]] = {}
            with ThreadPoolExecutor(
                max_workers=min(concurrency, len(batch_pages))
            ) as executor:
                futures = {
                    executor.submit(
                        scheduler.fetch_search,
                        _sample_parameters(row, current_page, per_page),
                    ): current_page
                    for current_page in batch_pages
                }
                for future in as_completed(futures):
                    payloads[futures[future]] = future.result()
        except DailyBudgetExhausted as error:
            _save_discovery_checkpoint(
                connection,
                row,
                max(sample_total, 0),
                retrieved_rows,
                pages,
                page,
                False,
                "daily_budget_exhausted",
                safe_provider_error(error),
            )
            raise
        batch_rows = 0
        for current_page in sorted(payloads):
            payload = payloads[current_page]
            results = payload.get("results")
            meta = payload.get("meta")
            if not isinstance(results, list) or not isinstance(meta, dict):
                raise ValueError(
                    "Malformed OpenAlex deterministic sample response"
                )
            if sample_total < 0:
                sample_total = int(meta.get("count") or 0)
            _save_discovery_page(
                connection,
                row,
                current_page,
                results,
            )
            batch_rows += len(results)
        retrieved_rows += batch_rows
        pages += len(payloads)
        expected_pages = math.ceil(sample_total / per_page)
        complete = (
            batch_rows == 0
            or max(payloads) >= expected_pages
            or retrieved_rows >= sample_total
        )
        next_page = max(payloads) + 1
        _save_discovery_checkpoint(
            connection,
            row,
            sample_total,
            retrieved_rows,
            pages,
            next_page,
            complete,
            (
                ""
                if complete
                else (
                    "review_capacity_loaded"
                    if maximum_page is not None
                    and next_page > maximum_page
                    else ""
                )
            ),
        )
        if complete:
            break
        if maximum_page is not None and next_page > maximum_page:
            break
        page = next_page
    final = connection.execute(
        """
        SELECT * FROM discovery_query_runs
        WHERE discovery_query_id = ?
        """,
        (row["discovery_query_id"],),
    ).fetchone()
    if final is None:
        raise RuntimeError("Discovery retrieval did not persist a checkpoint")
    return dict(final)


def _mark_bootstrap_saturation_stage(
    connection: sqlite3.Connection,
) -> Dict[str, Any]:
    rows = connection.execute(
        """
        SELECT q.discovery_query_id, r.complete, r.retrieved_rows,
               r.unique_hits, r.pages, r.stopped_reason
        FROM discovery_queries q
        LEFT JOIN discovery_query_runs r USING(discovery_query_id)
        WHERE q.status = 'active'
        ORDER BY q.discovery_query_id
        """
    ).fetchall()
    complete = bool(rows) and all(row["complete"] for row in rows)
    unique_records = connection.execute(
        "SELECT COUNT(DISTINCT record_key) FROM discovery_hits"
    ).fetchone()[0]
    details = {
        "retrieval_design": "deterministic_stratified_evidence_saturation",
        "active_strata": len(rows),
        "complete_strata": sum(int(row["complete"] or 0) for row in rows),
        "retrieved_rows_before_deduplication": sum(
            int(row["retrieved_rows"] or 0) for row in rows
        ),
        "unique_records": unique_records,
        "pages": sum(int(row["pages"] or 0) for row in rows),
        "broad_query_exhaustive": False,
    }
    set_stage(
        connection,
        "bootstrap_retrieval_complete",
        "complete" if complete else "ready",
        details,
    )
    bootstrap_query = connection.execute(
        """
        SELECT query_hash FROM physical_queries
        WHERE physical_query_id =
              'B0001_DOMAIN_AGNOSTIC_BOOTSTRAP__P001'
        """
    ).fetchone()
    if bootstrap_query is not None:
        inventory_row = connection.execute(
            """
            SELECT reported_total FROM query_runs
            WHERE provider = 'OpenAlex'
              AND physical_query_id =
                  'B0001_DOMAIN_AGNOSTIC_BOOTSTRAP__P001'
              AND run_role IN ('bootstrap_inventory', 'bootstrap')
              AND reported_total IS NOT NULL
            ORDER BY CASE run_role
                         WHEN 'bootstrap_inventory' THEN 0
                         ELSE 1
                     END
            LIMIT 1
            """
        ).fetchone()
        broad_total = int(inventory_row[0]) if inventory_row else 0
        connection.execute(
            """
            INSERT INTO query_runs(
                provider, physical_query_id, run_role, query_hash,
                reported_total, retrieved_rows, unique_hits, pages,
                next_cursor, complete, stopped_reason, error, updated_at
            ) VALUES (
                'OpenAlex',
                'B0001_DOMAIN_AGNOSTIC_BOOTSTRAP__P001',
                'bootstrap', ?, ?, ?, ?, ?, '', ?,
                'evidence_saturation_sample', '', ?
            )
            ON CONFLICT(provider, physical_query_id, run_role) DO UPDATE SET
                retrieved_rows = excluded.retrieved_rows,
                unique_hits = excluded.unique_hits,
                pages = excluded.pages,
                complete = excluded.complete,
                stopped_reason = excluded.stopped_reason,
                error = '',
                updated_at = excluded.updated_at
            """,
            (
                bootstrap_query["query_hash"],
                broad_total,
                details["retrieved_rows_before_deduplication"],
                unique_records,
                details["pages"],
                int(complete),
                utc_now(),
            ),
        )
    connection.commit()
    return details


def retrieve_discovery_samples(
    connection: sqlite3.Connection,
    maximum_queries: int | None = None,
    update_bootstrap_stage: bool = True,
) -> Dict[str, Any]:
    """Retrieve all frozen deterministic strata, stopping at daily credit."""
    if connection.execute(
        "SELECT COUNT(*) FROM discovery_queries WHERE status = 'active'"
    ).fetchone()[0] == 0:
        derive_discovery_queries(connection)
    protocol = read_json(SATURATION_PROTOCOL_PATH)
    per_page = int(protocol["api"]["per_page"])
    concurrency = int(protocol["api"]["concurrency"])
    scheduler = OpenAlexBudgetScheduler(connection)
    rows = connection.execute(
        """
        SELECT q.*
        FROM discovery_queries AS q
        LEFT JOIN discovery_query_runs AS r
          ON r.discovery_query_id = q.discovery_query_id
        WHERE q.status = 'active'
          AND COALESCE(r.complete, 0) = 0
        ORDER BY q.query_role, q.discovery_query_id
        """
    ).fetchall()
    if maximum_queries is not None:
        rows = rows[:maximum_queries]
    processed: List[Dict[str, Any]] = []
    budget_message = ""
    for row in rows:
        try:
            processed.append(
                _retrieve_one_discovery_query(
                    connection,
                    row,
                    scheduler,
                    per_page,
                    concurrency,
                )
            )
        except DailyBudgetExhausted as error:
            budget_message = str(error)
            break
    if update_bootstrap_stage:
        details = _mark_bootstrap_saturation_stage(connection)
    else:
        details = {
            "retrieval_design": (
                "deterministic_stratified_evidence_saturation"
            ),
            "active_strata": len(rows),
            "complete_strata": int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM discovery_query_runs r
                    JOIN discovery_queries q USING(discovery_query_id)
                    WHERE q.status = 'active' AND r.complete = 1
                    """
                ).fetchone()[0]
            ),
        }
    details["queries_visited_this_run"] = len(processed)
    details["budget_message"] = budget_message
    return details


def rebuild_discovery_review_ranks(
    connection: sqlite3.Connection,
) -> Dict[str, Any]:
    """Apply the frozen hash order without overwriting reviewed work."""
    reviewed_decisions = connection.execute(
        """
        SELECT COUNT(DISTINCT s.record_key)
        FROM screening_decisions s
        JOIN discovery_hits h ON h.record_key = s.record_key
        """
    ).fetchone()[0]
    if reviewed_decisions:
        raise RuntimeError(
            "Review ranks cannot be rebuilt after discovery decisions exist"
        )
    connection.execute(
        """
        WITH ranked AS (
            SELECT discovery_query_id, record_key,
                   ROW_NUMBER() OVER (
                       PARTITION BY discovery_query_id
                       ORDER BY selection_hash, record_key
                   ) AS new_rank
            FROM discovery_hits
        )
        UPDATE discovery_hits
        SET review_rank = (
            SELECT ranked.new_rank
            FROM ranked
            WHERE ranked.discovery_query_id =
                      discovery_hits.discovery_query_id
              AND ranked.record_key = discovery_hits.record_key
        ),
            review_round = 0,
            review_status = 'unassigned'
        """
    )
    connection.execute("DELETE FROM discovery_review_rounds")
    connection.commit()
    return {
        "ranked_hits": connection.execute(
            "SELECT COUNT(*) FROM discovery_hits WHERE review_rank > 0"
        ).fetchone()[0],
        "selection_order": "sha256_then_record_key",
    }


def _rank_unreviewed_new_queries(connection: sqlite3.Connection) -> int:
    mixed = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT discovery_query_id
            FROM discovery_hits
            GROUP BY discovery_query_id
            HAVING SUM(review_rank = 0) > 0
               AND SUM(review_rank > 0) > 0
        )
        """
    ).fetchone()[0]
    if mixed:
        raise RuntimeError(
            "A previously ranked discovery query received new rows; "
            "freeze a new sampling version before reassignment"
        )
    connection.execute(
        """
        WITH ranked AS (
            SELECT discovery_query_id, record_key,
                   ROW_NUMBER() OVER (
                       PARTITION BY discovery_query_id
                       ORDER BY selection_hash, record_key
                   ) AS new_rank
            FROM discovery_hits
            WHERE review_rank = 0
        )
        UPDATE discovery_hits
        SET review_rank = (
            SELECT ranked.new_rank
            FROM ranked
            WHERE ranked.discovery_query_id =
                      discovery_hits.discovery_query_id
              AND ranked.record_key = discovery_hits.record_key
        )
        WHERE review_rank = 0
        """
    )
    return int(connection.total_changes)


def ensure_formal_review_capacity(
    connection: sqlite3.Connection,
    iteration: int,
) -> Dict[str, int]:
    """Lazily fetch only formal-pool pages needed by the next review slice."""
    rows = connection.execute(
        """
        SELECT q.*, COALESCE(r.pages, 0) AS loaded_pages
        FROM discovery_queries q
        LEFT JOIN discovery_query_runs r USING(discovery_query_id)
        WHERE q.query_role = 'formal_search_family'
          AND q.status = 'active'
        ORDER BY q.discovery_query_id
        """
    ).fetchall()
    if not rows:
        return {"formal_queries": 0, "target_page": 0, "queries_extended": 0}
    protocol = read_json(SATURATION_PROTOCOL_PATH)
    width = int(
        protocol["sampling"]["records_per_stratum_by_role"].get(
            "formal_search_family",
            protocol["sampling"]["records_per_stratum_per_review_round"],
        )
    )
    offset_row = connection.execute(
        """
        SELECT value FROM metadata
        WHERE key = 'formal_review_rank_offset'
        """
    ).fetchone()
    offset = int(offset_row["value"]) if offset_row is not None else 0
    relative_last_rank = max(0, iteration * width - offset)
    per_page = int(protocol["api"]["per_page"])
    target_page = max(1, math.ceil(relative_last_rank / per_page))
    needs = [
        row for row in rows if int(row["loaded_pages"]) < target_page
    ]
    if not needs:
        return {
            "formal_queries": len(rows),
            "target_page": target_page,
            "queries_extended": 0,
        }
    scheduler = OpenAlexBudgetScheduler(connection)
    concurrency = int(protocol["api"]["concurrency"])
    database_path = Path(
        str(
            connection.execute(
                "PRAGMA database_list"
            ).fetchone()["file"]
        )
    )

    def extend_query(discovery_query_id: str) -> Dict[str, Any]:
        worker_connection = connect(database_path)
        try:
            worker_row = worker_connection.execute(
                """
                SELECT q.*, COALESCE(r.pages, 0) AS loaded_pages
                FROM discovery_queries q
                LEFT JOIN discovery_query_runs r
                  USING(discovery_query_id)
                WHERE q.discovery_query_id = ?
                """,
                (discovery_query_id,),
            ).fetchone()
            if worker_row is None:
                raise RuntimeError(
                    f"Missing formal pool: {discovery_query_id}"
                )
            return _retrieve_one_discovery_query(
                worker_connection,
                worker_row,
                scheduler,
                per_page,
                concurrency,
                maximum_page=target_page,
            )
        finally:
            worker_connection.close()

    first_error: Exception | None = None
    worker_count = min(
        concurrency,
        len(scheduler.keys),
        len(needs),
    )
    with ThreadPoolExecutor(
        max_workers=worker_count
    ) as executor:
        futures = {
            executor.submit(
                extend_query,
                str(row["discovery_query_id"]),
            ): str(row["discovery_query_id"])
            for row in needs
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                if first_error is None:
                    first_error = error
    if first_error is not None:
        raise first_error
    for row in rows:
        discovery_id = str(row["discovery_query_id"])
        if not discovery_id.startswith("FS_"):
            continue
        physical_id = discovery_id[3:]
        physical = connection.execute(
            """
            SELECT 1 FROM physical_queries
            WHERE physical_query_id = ?
            """,
            (physical_id,),
        ).fetchone()
        if physical is None:
            continue
        connection.execute(
            """
            INSERT OR IGNORE INTO query_hits(
                provider, physical_query_id, run_role, record_key, rank
            )
            SELECT 'OpenAlex', ?, 'formal', record_key, sample_rank
            FROM discovery_hits
            WHERE discovery_query_id = ?
            """,
            (physical_id, discovery_id),
        )
        current = connection.execute(
            """
            SELECT retrieved_rows, unique_hits, pages
            FROM discovery_query_runs
            WHERE discovery_query_id = ?
            """,
            (discovery_id,),
        ).fetchone()
        if current is not None:
            connection.execute(
                """
                UPDATE query_runs
                SET retrieved_rows = ?, unique_hits = ?, pages = ?,
                    updated_at = ?
                WHERE provider = 'OpenAlex'
                  AND physical_query_id = ?
                  AND run_role = 'formal'
                """,
                (
                    int(current["retrieved_rows"]),
                    int(current["unique_hits"]),
                    int(current["pages"]),
                    utc_now(),
                    physical_id,
                ),
            )
    connection.commit()
    return {
        "formal_queries": len(rows),
        "target_page": target_page,
        "queries_extended": len(needs),
    }


def _deduplicate_cross_round_assignments(
    connection: sqlite3.Connection,
) -> Dict[str, int]:
    """Keep each paper in its earliest review round across all strata."""
    duplicate_records = connection.execute(
        """
        SELECT record_key, MIN(review_round) AS first_round
        FROM discovery_hits
        WHERE review_round > 0
        GROUP BY record_key
        HAVING COUNT(DISTINCT review_round) > 1
        """
    ).fetchall()
    normalized_hits = 0
    for row in duplicate_records:
        cursor = connection.execute(
            """
            UPDATE discovery_hits
            SET review_round = 0,
                review_status = 'duplicate_prior_round'
            WHERE record_key = ?
              AND review_round > ?
            """,
            (row["record_key"], int(row["first_round"])),
        )
        normalized_hits += max(cursor.rowcount, 0)
    if duplicate_records:
        connection.execute(
            """
            UPDATE discovery_review_rounds
            SET assigned_records = (
                SELECT COUNT(DISTINCT h.record_key)
                FROM discovery_hits h
                WHERE h.review_round = discovery_review_rounds.iteration
            )
            """
        )
    return {
        "records": len(duplicate_records),
        "hits": normalized_hits,
    }


def assign_discovery_round(
    connection: sqlite3.Connection,
    iteration: int,
) -> Dict[str, Any]:
    """Assign the next fixed rank slice in every discovery stratum."""
    if iteration < 1:
        raise ValueError("Discovery iteration must be at least one")
    capacity = ensure_formal_review_capacity(connection, iteration)
    saturation_phase = (
        "formal_indicator_discovery"
        if connection.execute(
            """
            SELECT 1 FROM discovery_queries
            WHERE query_role = 'formal_search_family'
              AND status = 'active'
            LIMIT 1
            """
        ).fetchone()
        is not None
        else "search_frame_discovery"
    )
    frozen_phase = connection.execute(
        """
        SELECT iteration, stop_basis FROM discovery_review_rounds
        WHERE saturation_phase = ?
          AND decision = 'freeze'
        ORDER BY iteration DESC LIMIT 1
        """,
        (saturation_phase,),
    ).fetchone()
    if frozen_phase is not None and iteration > int(
        frozen_phase["iteration"]
    ):
        raise RuntimeError(
            f"{saturation_phase} already froze at iteration "
            f"{frozen_phase['iteration']} with stop basis "
            f"{frozen_phase['stop_basis']}; a later round cannot be assigned"
        )
    protocol = read_json(SATURATION_PROTOCOL_PATH)
    default_width = int(
        protocol["sampling"]["records_per_stratum_per_review_round"]
    )
    role_widths = protocol["sampling"].get(
        "records_per_stratum_by_role",
        {},
    )
    duplicate_assignments = _deduplicate_cross_round_assignments(connection)
    if connection.execute(
        "SELECT COUNT(*) FROM discovery_hits WHERE review_rank = 0"
    ).fetchone()[0]:
        reviewed_decisions = connection.execute(
            """
            SELECT COUNT(DISTINCT s.record_key)
            FROM screening_decisions s
            JOIN discovery_hits h ON h.record_key = s.record_key
            """
        ).fetchone()[0]
        if reviewed_decisions:
            _rank_unreviewed_new_queries(connection)
        else:
            rebuild_discovery_review_ranks(connection)
    ranges: Dict[str, List[int]] = {}
    for role_row in connection.execute(
        """
        SELECT DISTINCT query_role FROM discovery_queries
        WHERE status IN ('active', 'network')
        """
    ):
        role = str(role_row["query_role"])
        width = int(role_widths.get(role, default_width))
        first_rank = (iteration - 1) * width + 1
        last_rank = iteration * width
        ranges[role] = [first_rank, last_rank]
        connection.execute(
            """
            UPDATE discovery_hits
            SET review_round = ?
            WHERE review_round = 0
              AND record_key NOT IN (
                  SELECT record_key
                  FROM discovery_hits
                  WHERE review_round > 0
              )
              AND review_rank BETWEEN ? AND ?
              AND discovery_query_id IN (
                  SELECT discovery_query_id FROM discovery_queries
                  WHERE status IN ('active', 'network')
                    AND query_role = ?
              )
            """,
            (iteration, first_rank, last_rank, role),
        )
    assigned = connection.execute(
        """
        SELECT COUNT(DISTINCT record_key)
        FROM discovery_hits
        WHERE review_round = ?
        """,
        (iteration,),
    ).fetchone()[0]
    connection.execute(
        """
        INSERT INTO discovery_review_rounds(
            iteration, saturation_phase,
            batch_first_rank, batch_last_rank,
            assigned_records, fully_reviewed,
            new_nonredundant_english_terms,
            new_canonical_indicator_families,
            consecutive_zero_rounds, reviewer_role, decision,
            notes, reviewed_at
        ) VALUES (?, ?, ?, ?, ?, 0, -1, -1, 0,
                  'SYSTEM', 'pending', '', ?)
        ON CONFLICT(iteration) DO UPDATE SET
            batch_first_rank = excluded.batch_first_rank,
            batch_last_rank = excluded.batch_last_rank,
            assigned_records = excluded.assigned_records,
            reviewed_at = excluded.reviewed_at
        """,
        (
            iteration,
            saturation_phase,
            min(value[0] for value in ranges.values()),
            max(value[1] for value in ranges.values()),
            assigned,
            utc_now(),
        ),
    )
    connection.commit()
    return {
        "iteration": iteration,
        "saturation_phase": saturation_phase,
        "formal_capacity": capacity,
        "rank_ranges_by_role": ranges,
        "unique_records": assigned,
        "cross_round_duplicates_normalized": duplicate_assignments,
    }


def _round_records(
    connection: sqlite3.Connection,
    iteration: int,
) -> List[sqlite3.Row]:
    return connection.execute(
        """
        SELECT r.*, GROUP_CONCAT(DISTINCT h.discovery_query_id)
                      AS discovery_query_ids
        FROM records r
        JOIN discovery_hits h ON h.record_key = r.record_key
        WHERE h.review_round = ?
        GROUP BY r.record_key
        ORDER BY r.record_key
        """,
        (iteration,),
    ).fetchall()


def export_discovery_screening(
    connection: sqlite3.Connection,
    iteration: int,
    reviewer_role: str,
    output_path: Path,
) -> int:
    """Export one blind-compatible screening worksheet for a round."""
    role = reviewer_role.upper()
    if role not in {"AI", "H1", "H2"}:
        raise ValueError("reviewer_role must be AI, H1, or H2")
    fields = list(screening.SCREENING_FIELDS) + [
        "review_round",
        "discovery_query_ids",
    ]
    if role == "H2":
        fields.extend(
            [
                "h2_review_reason",
                "ai_language_judgment",
                "ai_decision",
                "ai_exclusion_reason",
                "ai_evidence_span",
                "ai_notes",
                "h1_language_judgment",
                "h1_decision",
                "h1_exclusion_reason",
                "h1_evidence_span",
                "h1_notes",
            ]
        )

    def rows() -> Iterable[Dict[str, Any]]:
        for record in _round_records(connection, iteration):
            decisions: Dict[str, sqlite3.Row] = {}
            h2_reason = ""
            if role == "H2":
                decisions = {
                    str(item["reviewer_role"]): item
                    for item in connection.execute(
                        """
                        SELECT * FROM screening_decisions
                        WHERE record_key = ?
                          AND reviewer_role IN ('AI', 'H1')
                        """,
                        (record["record_key"],),
                    )
                }
                if "AI" not in decisions or "H1" not in decisions:
                    raise RuntimeError(
                        "H2 export requires complete AI and H1 screening"
                    )
                required, h2_reason = screening._h2_requirement(
                    record,
                    decisions["AI"],
                    decisions["H1"],
                )
                if not required:
                    continue
            output: Dict[str, Any] = {
                "record_key": record["record_key"],
                "doi": record["doi"],
                "title": record["title"],
                "abstract": record["abstract"],
                "openalex_language": record["language"],
                "publication_year": record["publication_year"],
                "work_type": record["work_type"],
                "reviewer_role": role,
                "language_judgment": "",
                "language_evidence": "",
                "decision": "",
                "exclusion_reason": "",
                "evidence_span": "",
                "notes": "",
                "review_round": iteration,
                "discovery_query_ids": record["discovery_query_ids"],
            }
            if role == "H2":
                ai = decisions["AI"]
                h1 = decisions["H1"]
                output.update(
                    {
                        "h2_review_reason": h2_reason,
                        "ai_language_judgment": ai["language_judgment"],
                        "ai_decision": ai["decision"],
                        "ai_exclusion_reason": ai["exclusion_reason"],
                        "ai_evidence_span": ai["evidence_span"],
                        "ai_notes": ai["notes"],
                        "h1_language_judgment": h1[
                            "language_judgment"
                        ],
                        "h1_decision": h1["decision"],
                        "h1_exclusion_reason": h1["exclusion_reason"],
                        "h1_evidence_span": h1["evidence_span"],
                        "h1_notes": h1["notes"],
                    }
                )
            yield output

    return write_csv_iter(output_path, rows(), fields)


def finalize_discovery_screening(
    connection: sqlite3.Connection,
    iteration: int,
) -> Dict[str, Any]:
    """Finalize one discovery round without treating AI as a human coder."""
    records = _round_records(connection, iteration)
    missing: List[str] = []
    counts = {"include": 0, "exclude": 0, "h2_required": 0}
    for record in records:
        decisions = {
            str(item["reviewer_role"]): item
            for item in connection.execute(
                "SELECT * FROM screening_decisions WHERE record_key = ?",
                (record["record_key"],),
            )
        }
        if "AI" not in decisions or "H1" not in decisions:
            missing.append(f"{record['record_key']}:AI/H1")
            continue
        ai = decisions["AI"]
        h1 = decisions["H1"]
        required, _ = screening._h2_requirement(record, ai, h1)
        final = ai
        if required:
            counts["h2_required"] += 1
            if "H2" not in decisions:
                missing.append(f"{record['record_key']}:H2")
                continue
            final = decisions["H2"]
        decision = str(final["decision"])
        language = str(final["language_judgment"])
        if decision == "uncertain" or language == "uncertain":
            missing.append(f"{record['record_key']}:UNCERTAIN")
            continue
        if language == "non_en":
            decision = "exclude"
        counts[decision] += 1
        connection.execute(
            """
            UPDATE discovery_hits
            SET review_status = ?
            WHERE record_key = ? AND review_round = ?
            """,
            (decision, record["record_key"], iteration),
        )
    if missing:
        connection.rollback()
        raise RuntimeError(
            "Discovery screening is incomplete: " + ", ".join(missing[:25])
        )
    connection.execute(
        """
        UPDATE discovery_review_rounds
        SET fully_reviewed = 1, reviewed_at = ?
        WHERE iteration = ?
        """,
        (utc_now(), iteration),
    )
    connection.commit()
    return {"iteration": iteration, "records": len(records), **counts}


def export_discovery_extraction(
    connection: sqlite3.Connection,
    iteration: int,
    output_path: Path,
    extractor_role: str = "H1",
) -> int:
    """Export included papers for source-preserving term/indicator extraction."""
    role = extractor_role.upper()
    if role not in {"AI", "H1"}:
        raise ValueError("extractor_role must be AI or H1")

    def rows() -> Iterable[Dict[str, Any]]:
        for record in connection.execute(
            """
            SELECT DISTINCT r.*
            FROM records r
            JOIN discovery_hits h ON h.record_key = r.record_key
            WHERE h.review_round = ? AND h.review_status = 'include'
            ORDER BY r.record_key
            """,
            (iteration,),
        ):
            yield {
                "record_key": record["record_key"],
                "doi": record["doi"],
                "title": record["title"],
                "abstract": record["abstract"],
                "review_round": iteration,
                "item_type": "term|indicator_candidate",
                "verbatim_name": "",
                "location": "title|abstract",
                "evidence_span": "",
                "proposed_role": "",
                "status": "active",
                "exclusion_reason": "",
                "extractor_role": role,
                "canonical_family_label": "",
                "h1_decision": "pending",
                "h2_decision": "pending",
                "record_extraction_complete": "false",
                "no_relevant_items": "false",
                "review_notes": "",
            }

    return write_csv_iter(output_path, rows(), DISCOVERY_EXTRACTION_FIELDS)


def import_discovery_extraction(
    connection: sqlite3.Connection,
    input_path: Path,
) -> Dict[str, int]:
    """Import exact English term spans and provisional indicator names."""
    extraction_rows = list(iter_csv(input_path))
    assisted_roles = {
        str(row.get("extractor_role") or "").strip().upper()
        for row in extraction_rows
        if str(row.get("extractor_role") or "").strip()
    }
    if any(
        str(row.get("draft_method") or "").strip()
        for row in extraction_rows
    ):
        if len(assisted_roles) != 1:
            raise ValueError(
                "One assisted discovery extraction must declare one role"
            )
        assert_registered_review_attestation(
            connection,
            input_path,
            next(iter(assisted_roles)),
        )
    snapshot_path = snapshot_import_file(
        connection,
        input_path,
        "discovery_term_indicator_extraction",
    )
    counts = {
        "terms": 0,
        "indicator_candidates": 0,
        "completed_record_reviews": 0,
    }
    review_keys: set[tuple[str, int, str]] = set()
    review_no_item_states: Dict[tuple[str, int, str], bool] = {}
    extracted_item_keys: set[tuple[str, int, str, str, str]] = set()
    submission_role = ""
    for row in iter_csv(snapshot_path):
        name = str(row.get("verbatim_name") or "").strip()
        record_key = str(row.get("record_key") or "").strip()
        if not record_key:
            continue
        record = connection.execute(
            "SELECT * FROM records WHERE record_key = ?",
            (record_key,),
        ).fetchone()
        if record is None:
            raise ValueError(f"Unknown discovery record: {record_key}")
        for source_field in ("doi", "title", "abstract"):
            supplied = str(row.get(source_field) or "")
            stored = str(record[source_field] or "")
            if supplied and supplied != stored:
                raise ValueError(
                    "Discovery extraction must preserve the frozen source "
                    f"{source_field}: {record_key}"
                )
        record_language = str(
            record["language"] or "unknown"
        ).strip().casefold()
        h2_language_approval = connection.execute(
            """
            SELECT 1 FROM screening_decisions
            WHERE record_key = ? AND reviewer_role = 'H2'
              AND language_judgment = 'en' AND decision = 'include'
            """,
            (record_key,),
        ).fetchone()
        if record_language != "en" and not (
            record_language in {"", "unknown"}
            and h2_language_approval is not None
        ):
            raise ValueError(
                "Active discovery extraction requires OpenAlex English or "
                f"H2-confirmed English when language is unknown: {record_key}"
            )
        iteration = int(row.get("review_round") or 0)
        included = connection.execute(
            """
            SELECT 1 FROM discovery_hits
            WHERE record_key = ? AND review_round = ?
              AND review_status = 'include'
            LIMIT 1
            """,
            (record_key, iteration),
        ).fetchone()
        if included is None:
            raise ValueError(
                "Extraction is allowed only for a finalized included record: "
                f"{record_key}/round={iteration}"
            )
        extractor_role = str(
            row.get("extractor_role") or ""
        ).strip().upper()
        if extractor_role not in {"AI", "H1"}:
            raise ValueError(
                f"Invalid discovery extractor role: {extractor_role!r}"
            )
        if submission_role and extractor_role != submission_role:
            raise ValueError(
                "One discovery-extraction import cannot mix AI and H1 roles"
            )
        submission_role = extractor_role
        adjudicated_candidate = connection.execute(
            """
            SELECT 1 FROM discovery_indicator_candidates
            WHERE record_key = ? AND review_round = ?
              AND h2_decision != 'pending'
            LIMIT 1
            """,
            (record_key, iteration),
        ).fetchone()
        if adjudicated_candidate is not None:
            raise ValueError(
                "Primary discovery extraction is frozen after H2 indicator "
                f"adjudication: {record_key}/round={iteration}"
            )
        extraction_complete = parse_bool(
            row.get("record_extraction_complete"),
            "record_extraction_complete",
        )
        no_relevant_items = parse_bool(
            row.get("no_relevant_items"),
            "no_relevant_items",
        )
        if not extraction_complete:
            raise ValueError(
                f"Imported extraction row is not complete: {record_key}"
            )
        if name and no_relevant_items:
            raise ValueError(
                f"A record with an extracted item cannot be no-items: "
                f"{record_key}"
            )
        if not name and not no_relevant_items:
            raise ValueError(
                "A completed blank extraction row must set "
                f"no_relevant_items=true: {record_key}"
            )
        status = str(row.get("status") or "").strip().casefold()
        if status != "active":
            raise ValueError(
                f"Completed discovery extraction must be active: {record_key}"
            )
        if str(row.get("exclusion_reason") or "").strip():
            raise ValueError(
                "Completed active discovery extraction cannot have an "
                f"exclusion reason: {record_key}"
            )
        review_key = (record_key, iteration, extractor_role)
        prior_no_items = review_no_item_states.get(review_key)
        if (
            prior_no_items is not None
            and prior_no_items != no_relevant_items
        ):
            raise ValueError(
                "One completed extraction cannot mix no-items and extracted "
                f"item rows: {record_key}/round={iteration}/{extractor_role}"
            )
        review_no_item_states[review_key] = no_relevant_items
        connection.execute(
            """
            INSERT INTO discovery_extraction_reviews(
                record_key, review_round, reviewer_role,
                extraction_complete, no_relevant_items, notes, reviewed_at
            ) VALUES (?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(record_key, review_round, reviewer_role) DO UPDATE SET
                extraction_complete = 1,
                no_relevant_items = excluded.no_relevant_items,
                notes = excluded.notes,
                reviewed_at = excluded.reviewed_at
            """,
            (
                record_key,
                iteration,
                extractor_role,
                int(no_relevant_items),
                str(row.get("review_notes") or "").strip(),
                utc_now(),
            ),
        )
        review_keys.add(review_key)
        if not name:
            continue
        evidence_span = str(row.get("evidence_span") or "").strip()
        if not evidence_span:
            raise ValueError(f"Evidence span is required for {name!r}")
        if name.casefold() not in evidence_span.casefold():
            raise ValueError(
                "Evidence span must contain the verbatim name: "
                f"{record_key}/{name}"
            )
        item_type = str(row.get("item_type") or "").strip().casefold()
        location = str(row.get("location") or "").strip().casefold()
        if location not in {"title", "abstract"}:
            raise ValueError(
                "Discovery extraction is limited to title/abstract evidence; "
                f"invalid location={location!r}"
            )
        source_text = str(record[location] or "")
        if evidence_span.casefold() not in source_text.casefold():
            raise ValueError(
                f"Evidence span is not an exact {location} substring: "
                f"{record_key}/{name}"
            )
        if name.casefold() not in source_text.casefold():
            raise ValueError(
                f"Verbatim name is not present in the {location}: "
                f"{record_key}/{name}"
            )
        proposed_role = str(
            row.get("proposed_role") or ""
        ).strip().casefold()
        if proposed_role not in DISCOVERY_PROPOSED_ROLES:
            raise ValueError(
                f"Invalid discovery proposed_role: {proposed_role!r}"
            )
        item_key = (
            record_key,
            iteration,
            item_type,
            name.casefold(),
            location,
        )
        if item_key in extracted_item_keys:
            raise ValueError(
                f"Duplicate discovery extraction item: {record_key}/{name}"
            )
        extracted_item_keys.add(item_key)
        if item_type == "term":
            coding._assert_not_frozen(connection)
            term_id = "TERM_" + sha256_bytes(
                f"{record_key}|{name}|{location}".encode("utf-8")
            )[:16].upper()
            normalized = normalize_term(name)
            existing_term = connection.execute(
                "SELECT * FROM raw_terms WHERE term_id = ?",
                (term_id,),
            ).fetchone()
            if existing_term is not None:
                term_is_coded = connection.execute(
                    "SELECT 1 FROM term_coding WHERE term_id = ? LIMIT 1",
                    (term_id,),
                ).fetchone()
                changed_evidence = (
                    str(existing_term["evidence_span"]) != evidence_span
                    or str(existing_term["proposed_role"])
                    != str(row.get("proposed_role") or "").strip()
                )
                if term_is_coded is not None and changed_evidence:
                    raise ValueError(
                        "Coded term evidence is immutable; record a new "
                        f"correction/version instead: {term_id}"
                    )
            connection.execute(
                """
                INSERT INTO raw_terms(
                    term_id, source_record_key, source_id, source_type,
                    source_language_status, source_language_evidence,
                    verbatim_term, normalized_term, match_key, location,
                    evidence_span, proposed_role, status, exclusion_reason
                ) VALUES (?, ?, ?, 'bootstrap_literature', 'en', ?, ?, ?,
                          ?, ?, ?, ?, 'active', '')
                ON CONFLICT(term_id) DO UPDATE SET
                    evidence_span = excluded.evidence_span,
                    proposed_role = excluded.proposed_role,
                    status = 'active',
                    exclusion_reason = ''
                """,
                (
                    term_id,
                    record_key,
                    record["doi"] or record["provider_id"],
                    (
                        f"OpenAlex language={record_language}; "
                        + (
                            "H2 language=en; "
                            if record_language != "en"
                            else ""
                        )
                        + f"{location} evidence verified"
                    ),
                    name,
                    normalized,
                    term_match_key(name),
                    location,
                    evidence_span,
                    proposed_role,
                ),
            )
            counts["terms"] += 1
        elif item_type == "indicator_candidate":
            h1_decision = str(
                row.get("h1_decision") or "pending"
            ).strip().casefold()
            if extractor_role == "H1" and h1_decision == "pending":
                h1_decision = "include"
            h2_decision = str(
                row.get("h2_decision") or "pending"
            ).strip().casefold()
            if h1_decision not in {"pending", "include", "exclude"}:
                raise ValueError(
                    f"Invalid H1 indicator decision: {h1_decision}"
                )
            if h2_decision not in {"pending", "include", "exclude"}:
                raise ValueError(
                    f"Invalid H2 indicator decision: {h2_decision}"
                )
            if h2_decision != "pending":
                raise ValueError(
                    "AI/H1 extraction cannot submit an H2 indicator "
                    f"decision: {record_key}/{name}"
                )
            if extractor_role == "AI" and h1_decision != "pending":
                raise ValueError(
                    "AI extraction cannot submit an H1 indicator decision: "
                    f"{record_key}/{name}"
                )
            family_label = str(
                row.get("canonical_family_label") or ""
            ).strip()
            if family_label:
                raise ValueError(
                    "Canonical indicator-family labels are restricted to "
                    f"the H2 adjudication import: {record_key}/{name}"
                )
            candidate_id = "DIC_" + sha256_bytes(
                f"{record_key}|{name}|{location}".encode("utf-8")
            )[:16].upper()
            connection.execute(
                """
                INSERT INTO discovery_indicator_candidates(
                    candidate_id, record_key, review_round, raw_name_en,
                    normalized_name, location, evidence_span, proposed_role,
                    extracted_by, h1_decision, h2_decision,
                    canonical_family_label, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'candidate')
                ON CONFLICT(candidate_id) DO UPDATE SET
                    evidence_span = excluded.evidence_span,
                    proposed_role = excluded.proposed_role,
                    extracted_by = excluded.extracted_by,
                    h1_decision = excluded.h1_decision
                """,
                (
                    candidate_id,
                    record_key,
                    iteration,
                    name,
                    normalize_term(name),
                    location,
                    evidence_span,
                    proposed_role,
                    extractor_role,
                    h1_decision,
                    h2_decision,
                    family_label,
                ),
            )
            counts["indicator_candidates"] += 1
        else:
            raise ValueError(f"Invalid discovery item_type: {item_type!r}")
    counts["completed_record_reviews"] = len(review_keys)
    if counts["terms"]:
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
            "new pre-freeze discovery terms changed the search frame",
        )
    elif counts["indicator_candidates"]:
        invalidate_stages(
            connection,
            (
                "indicators_extracted",
                "dimensions_derived",
                "features_selected",
                "audit_complete",
            ),
            "discovery indicator candidates changed",
        )
    else:
        invalidate_stages(
            connection,
            ("audit_complete",),
            "discovery extraction review completion changed",
        )
    connection.commit()
    return counts


def export_discovery_indicator_adjudication(
    connection: sqlite3.Connection,
    iteration: int,
    output_path: Path,
) -> int:
    """Export H1-retained indicator names for independent H2 adjudication."""

    def rows() -> Iterable[Dict[str, Any]]:
        for row in connection.execute(
            """
            SELECT c.*, r.doi, r.title, r.abstract
            FROM discovery_indicator_candidates c
            JOIN records r USING(record_key)
            WHERE c.review_round = ?
              AND c.status = 'candidate'
              AND c.h1_decision = 'include'
            ORDER BY c.candidate_id
            """,
            (iteration,),
        ):
            yield {
                "candidate_id": row["candidate_id"],
                "record_key": row["record_key"],
                "review_round": row["review_round"],
                "doi": row["doi"],
                "title": row["title"],
                "abstract": row["abstract"],
                "raw_name_en": row["raw_name_en"],
                "location": row["location"],
                "evidence_span": row["evidence_span"],
                "proposed_role": row["proposed_role"],
                "extracted_by": row["extracted_by"],
                "h1_decision": row["h1_decision"],
                "h2_decision": "",
                "canonical_family_label": "",
                "adjudication_notes": "",
            }

    return write_csv_iter(
        output_path,
        rows(),
        DISCOVERY_INDICATOR_ADJUDICATION_FIELDS,
    )


def import_discovery_indicator_adjudication(
    connection: sqlite3.Connection,
    input_path: Path,
) -> int:
    """Import H2 decisions without allowing an unlabelled included family."""
    adjudication_rows = list(iter_csv(input_path))
    if any(
        str(row.get("draft_method") or "").strip()
        for row in adjudication_rows
    ):
        assert_registered_review_attestation(
            connection,
            input_path,
            "H2",
        )
    snapshot_path = snapshot_import_file(
        connection,
        input_path,
        "discovery_indicator_adjudication",
    )
    imported = 0
    for row in iter_csv(snapshot_path):
        candidate_id = str(row.get("candidate_id") or "").strip()
        decision = str(row.get("h2_decision") or "").strip().casefold()
        if not candidate_id or not decision:
            continue
        if decision not in {"include", "exclude"}:
            raise ValueError(f"Invalid H2 indicator decision: {decision}")
        candidate = connection.execute(
            """
            SELECT * FROM discovery_indicator_candidates
            WHERE candidate_id = ? AND h1_decision = 'include'
            """,
            (candidate_id,),
        ).fetchone()
        if candidate is None:
            raise ValueError(
                f"Unknown H1-included indicator candidate: {candidate_id}"
            )
        family = str(
            row.get("canonical_family_label") or ""
        ).strip()
        notes = str(row.get("adjudication_notes") or "").strip()
        if decision == "include" and not family:
            raise ValueError(
                f"H2-included candidate needs a family: {candidate_id}"
            )
        if decision == "exclude" and family:
            raise ValueError(
                f"H2-excluded candidate must leave family blank: "
                f"{candidate_id}"
            )
        if not notes:
            raise ValueError(
                f"H2 indicator adjudication needs notes: {candidate_id}"
            )
        connection.execute(
            """
            UPDATE discovery_indicator_candidates
            SET h2_decision = ?, canonical_family_label = ?,
                adjudication_notes = ?
            WHERE candidate_id = ?
            """,
            (decision, family, notes, candidate_id),
        )
        imported += 1
    if imported:
        invalidate_stages(
            connection,
            (
                "indicators_extracted",
                "dimensions_derived",
                "features_selected",
                "audit_complete",
            ),
            "H2 discovery indicator-family adjudication changed",
        )
    connection.commit()
    return imported


def _adjudicated_term_family_origins(
    connection: sqlite3.Connection,
) -> tuple[Dict[str, int], List[str]]:
    """Return each retained term family's first discovery round."""
    origins: Dict[str, int] = {}
    unresolved: List[str] = []
    for term in connection.execute(
        """
        SELECT t.*,
               COALESCE(
                   (
                       SELECT MIN(h.review_round)
                       FROM discovery_hits h
                       WHERE h.record_key = t.source_record_key
                         AND h.review_round > 0
                         AND h.review_status = 'include'
                   ),
                   0
               ) AS origin_round
        FROM raw_terms t
        WHERE t.status = 'active'
        ORDER BY t.term_id
        """
    ):
        codes = {
            str(row["coder_role"]): row
            for row in connection.execute(
                "SELECT * FROM term_coding WHERE term_id = ?",
                (term["term_id"],),
            )
        }
        if "AI" not in codes or "H1" not in codes:
            unresolved.append(f"{term['term_id']}:AI/H1")
            continue
        ai = codes["AI"]
        h1 = codes["H1"]
        both_exclude = (
            coding._coding_signature(ai) == coding._coding_signature(h1)
            and ai["decision"] == "exclude"
        )
        if both_exclude:
            continue
        final = codes.get("H2")
        if final is None:
            unresolved.append(f"{term['term_id']}:H2")
            continue
        if str(final["decision"]) == "exclude":
            continue
        label = normalize_term(str(final["term_family_label"]))
        if not label:
            unresolved.append(f"{term['term_id']}:EMPTY_FAMILY")
            continue
        origin_round = int(term["origin_round"])
        origins[label] = min(origins.get(label, origin_round), origin_round)
    return origins, unresolved


def _adjudicated_indicator_family_origins(
    connection: sqlite3.Connection,
) -> tuple[Dict[str, int], List[str]]:
    """Return each retained discovery indicator family's first round."""
    origins: Dict[str, int] = {}
    unresolved: List[str] = []
    for candidate in connection.execute(
        """
        SELECT * FROM discovery_indicator_candidates
        WHERE status = 'candidate'
        ORDER BY candidate_id
        """
    ):
        h1 = str(candidate["h1_decision"]).casefold()
        h2 = str(candidate["h2_decision"]).casefold()
        if h1 == "pending":
            unresolved.append(f"{candidate['candidate_id']}:H1")
            continue
        if h1 == "exclude":
            continue
        if h2 == "pending":
            unresolved.append(f"{candidate['candidate_id']}:H2")
            continue
        if h2 == "exclude":
            continue
        label = normalize_term(str(candidate["canonical_family_label"]))
        if not label:
            unresolved.append(
                f"{candidate['candidate_id']}:EMPTY_FAMILY"
            )
            continue
        origin_round = int(candidate["review_round"])
        origins[label] = min(origins.get(label, origin_round), origin_round)
    return origins, unresolved


def discovery_novelty_counts(
    connection: sqlite3.Connection,
    iteration: int,
) -> Dict[str, Any]:
    """Compute auditable dual novelty endpoints for one reviewed round."""
    included_records = int(
        connection.execute(
            """
            SELECT COUNT(DISTINCT record_key)
            FROM discovery_hits
            WHERE review_round = ? AND review_status = 'include'
            """,
            (iteration,),
        ).fetchone()[0]
    )
    completed_h1_extractions = int(
        connection.execute(
            """
            SELECT COUNT(DISTINCT record_key)
            FROM discovery_extraction_reviews
            WHERE review_round = ? AND reviewer_role = 'H1'
              AND extraction_complete = 1
            """,
            (iteration,),
        ).fetchone()[0]
    )
    if completed_h1_extractions != included_records:
        raise RuntimeError(
            "Every included record needs a completed H1 extraction "
            f"disposition: included={included_records}, "
            f"complete={completed_h1_extractions}"
        )
    term_origins, unresolved_terms = _adjudicated_term_family_origins(
        connection
    )
    indicator_origins, unresolved_indicators = (
        _adjudicated_indicator_family_origins(connection)
    )
    unresolved = [*unresolved_terms, *unresolved_indicators]
    if unresolved:
        raise RuntimeError(
            "Discovery novelty cannot be counted before adjudication: "
            + ", ".join(unresolved[:25])
        )
    new_term_labels = sorted(
        label for label, origin in term_origins.items() if origin == iteration
    )
    new_indicator_labels = sorted(
        label
        for label, origin in indicator_origins.items()
        if origin == iteration
    )
    return {
        "iteration": iteration,
        "included_records": included_records,
        "completed_h1_extractions": completed_h1_extractions,
        "new_terms": len(new_term_labels),
        "new_indicator_families": len(new_indicator_labels),
        "new_term_family_labels": new_term_labels,
        "new_indicator_family_labels": new_indicator_labels,
    }


def record_discovery_saturation(
    connection: sqlite3.Connection,
    iteration: int,
    new_terms: int,
    new_indicator_families: int,
    decision: str,
    notes: str,
    reviewer_role: str = "H2",
    protocol_deviation_amendment: Path | None = None,
) -> Dict[str, Any]:
    """Record H2's evidence-preserving discovery stop decision."""
    role = reviewer_role.upper()
    if role != "H2":
        raise ValueError("Only H2 may approve discovery saturation")
    if new_terms < 0 or new_indicator_families < 0:
        raise ValueError("Novelty counts cannot be negative")
    if decision not in {"continue", "freeze"}:
        raise ValueError("decision must be continue or freeze")
    row = connection.execute(
        "SELECT * FROM discovery_review_rounds WHERE iteration = ?",
        (iteration,),
    ).fetchone()
    if row is None or not bool(row["fully_reviewed"]):
        raise RuntimeError("The discovery round is not fully reviewed")
    computed = discovery_novelty_counts(connection, iteration)
    if new_terms != computed["new_terms"]:
        raise ValueError(
            "Submitted new-term count does not match adjudicated evidence: "
            f"submitted={new_terms}, computed={computed['new_terms']}"
        )
    if new_indicator_families != computed["new_indicator_families"]:
        raise ValueError(
            "Submitted indicator-family count does not match adjudicated "
            f"evidence: submitted={new_indicator_families}, "
            f"computed={computed['new_indicator_families']}"
        )
    previous = connection.execute(
        """
        SELECT consecutive_zero_rounds
        FROM discovery_review_rounds
        WHERE iteration < ? AND saturation_phase = ?
        ORDER BY iteration DESC LIMIT 1
        """,
        (iteration, row["saturation_phase"]),
    ).fetchone()
    prior_zero = int(previous[0]) if previous is not None else 0
    is_zero = new_terms == 0 and new_indicator_families == 0
    consecutive = prior_zero + 1 if is_zero else 0
    required = int(
        read_json(SATURATION_PROTOCOL_PATH)["sequential_review"][
            "minimum_consecutive_zero_novelty_rounds"
        ]
    )
    stop_basis = "not_applicable"
    amendment_id = ""
    amendment_sha256 = ""
    if decision == "freeze":
        if consecutive >= required:
            stop_basis = "preregistered_consecutive_dual_zero"
            if protocol_deviation_amendment is not None:
                raise ValueError(
                    "A protocol-deviation amendment is not permitted when "
                    "the preregistered dual-zero rule is satisfied"
                )
        else:
            if protocol_deviation_amendment is None:
                raise ValueError(
                    f"Freeze requires {required} consecutive dual-zero "
                    f"rounds; current={consecutive}. An explicit audited "
                    "protocol-deviation amendment is required to stop early."
                )
            supplied_path = protocol_deviation_amendment.resolve()
            expected_path = DISCOVERY_STOP_AMENDMENT_PATH.resolve()
            if supplied_path != expected_path:
                raise ValueError(
                    "Only the frozen round-12 pragmatic-stop amendment may "
                    "authorize this protocol deviation"
                )
            amendment = read_json(supplied_path)
            guards = amendment.get("integrity_guards", {})
            required_guards = (
                "actual_endpoint_counts_must_be_retained",
                "dual_zero_claim_requires_computed_dual_zero",
                "protocol_deviation_must_be_disclosed",
                "round12_decisions_must_not_be_changed_to_create_zero",
                "later_search_frame_discovery_rounds_must_not_be_started",
            )
            if (
                amendment.get("status") != "approved"
                or amendment.get("approved_by") != "project_owner"
                or amendment.get("scope") != row["saturation_phase"]
                or int(amendment.get("terminal_iteration", -1)) != iteration
                or amendment.get("replacement_stop_basis")
                != "retrospective_owner_pragmatic_stop"
                or not all(guards.get(key) is True for key in required_guards)
            ):
                raise ValueError(
                    "The protocol-deviation amendment is incomplete or does "
                    "not authorize this phase and iteration"
                )
            stop_basis = "retrospective_owner_pragmatic_stop"
            amendment_id = str(amendment["amendment_id"])
            amendment_sha256 = sha256_bytes(supplied_path.read_bytes())
    if not notes.strip():
        raise ValueError("H2 saturation notes are required")
    stored_notes = notes.strip()
    if stop_basis == "retrospective_owner_pragmatic_stop":
        stored_notes = (
            f"{stored_notes} [PROTOCOL_DEVIATION:{amendment_id}; "
            f"actual_new_terms={new_terms}; "
            "actual_new_indicator_families="
            f"{new_indicator_families}; dual_zero={str(is_zero).lower()}]"
        )
    connection.execute(
        """
        UPDATE discovery_review_rounds
        SET new_nonredundant_english_terms = ?,
            new_canonical_indicator_families = ?,
            consecutive_zero_rounds = ?, reviewer_role = 'H2',
            decision = ?, stop_basis = ?, protocol_amendment_id = ?,
            protocol_amendment_sha256 = ?, notes = ?, reviewed_at = ?
        WHERE iteration = ?
        """,
        (
            new_terms,
            new_indicator_families,
            consecutive,
            decision,
            stop_basis,
            amendment_id,
            amendment_sha256,
            stored_notes,
            utc_now(),
            iteration,
        ),
    )
    if decision == "freeze" and row["saturation_phase"] == (
        "formal_indicator_discovery"
    ):
        freeze_details = {
            "reason": "formal evidence-saturation branch frozen",
            "freeze_iteration": iteration,
            "stop_basis": stop_basis,
            "consecutive_dual_zero_rounds": consecutive,
            "sampling_scope": (
                "deterministic evidence-saturation pools; unassigned "
                "citation-network records were not treated as screened"
            ),
        }
        set_stage(connection, "literature_screened", "complete", freeze_details)
        set_stage(
            connection,
            "indicators_extracted",
            "complete",
            {
                **freeze_details,
                "reason": (
                    "formal evidence-saturation branch frozen with no new "
                    "included sources; H1/H2 indicator extraction retained"
                ),
            },
        )
    connection.commit()
    return {
        "iteration": iteration,
        "saturation_phase": row["saturation_phase"],
        "new_terms": new_terms,
        "new_indicator_families": new_indicator_families,
        "consecutive_zero_rounds": consecutive,
        "required_zero_rounds": required,
        "decision": decision,
        "stop_basis": stop_basis,
        "protocol_amendment_id": amendment_id,
        "protocol_amendment_sha256": amendment_sha256,
        "new_term_family_labels": computed["new_term_family_labels"],
        "new_indicator_family_labels": computed[
            "new_indicator_family_labels"
        ],
    }


def discovery_status(connection: sqlite3.Connection) -> Dict[str, Any]:
    """Summarize retrieval, review, and non-secret budget state."""
    return {
        "queries": {
            "active": connection.execute(
                """
                SELECT COUNT(*) FROM discovery_queries
                WHERE status = 'active'
                """
            ).fetchone()[0],
            "complete": connection.execute(
                """
                SELECT COUNT(*) FROM discovery_query_runs
                WHERE complete = 1
                """
            ).fetchone()[0],
        },
        "records": connection.execute(
            "SELECT COUNT(DISTINCT record_key) FROM discovery_hits"
        ).fetchone()[0],
        "review_rounds": [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM discovery_review_rounds ORDER BY iteration"
            )
        ],
        "latest_budget": [
            dict(row)
            for row in connection.execute(
                """
                SELECT b.* FROM api_budget_observations b
                JOIN (
                    SELECT key_slot, MAX(observation_id) AS observation_id
                    FROM api_budget_observations GROUP BY key_slot
                ) latest USING(key_slot, observation_id)
                ORDER BY key_slot
                """
            )
        ],
    }
