from __future__ import annotations

import json
import re
import sqlite3
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Sequence

import saturation
from common import json_hash, read_json, utc_now
from database import invalidate_stages, log_event, require_complete, set_stage
from providers import (
    query_definition_hash,
    retrieve_physical_query,
)
from screening import included_record_keys


PROTOCOL_PATH = Path(__file__).resolve().parent / "protocol_v3.json"


def formal_query_ids(connection: sqlite3.Connection) -> List[str]:
    """Return active provider requests from the frozen formal frame."""
    return [
        str(row[0])
        for row in connection.execute(
            """
            SELECT p.physical_query_id
            FROM physical_queries p
            JOIN logical_queries l USING(logical_query_id)
            WHERE p.status = 'active' AND l.status = 'active'
              AND l.logical_query_id LIKE 'L%'
            ORDER BY p.physical_query_id
            """
        )
    ]


def retrieve_formal_queries(
    connection: sqlite3.Connection,
    max_pages_per_query: int | None = None,
    fetcher: Any | None = None,
) -> Dict[str, Any]:
    """Build and retrieve one frozen deterministic pool per formal query."""
    require_complete(connection, ["search_frame_frozen"])
    if max_pages_per_query is not None:
        raise ValueError(
            "Formal evidence-saturation pools do not use an arbitrary "
            "page cutoff"
        )
    if fetcher is not None:
        raise ValueError(
            "Custom fetchers are unsupported for budget-aware formal pools"
        )
    query_ids = formal_query_ids(connection)
    if not query_ids:
        raise RuntimeError("The frozen frame contains no active queries")
    _register_formal_saturation_pools(connection, query_ids)
    next_iteration = int(
        connection.execute(
            """
            SELECT COALESCE(MAX(iteration), 0) + 1
            FROM discovery_review_rounds
            """
        ).fetchone()[0]
    )
    saturation.ensure_formal_review_capacity(connection, next_iteration)
    _map_formal_saturation_pools(connection, query_ids)
    details = mark_formal_retrieval_stage(connection)
    invalidate_stages(
        connection,
        (
            "literature_screened",
            "indicators_extracted",
            "dimensions_derived",
            "features_selected",
            "audit_complete",
        ),
        "formal evidence-saturation pools changed",
    )
    connection.commit()
    return details


def _formal_discovery_query_id(
    connection: sqlite3.Connection,
    physical_query_id: str,
) -> str:
    """Return a frame-versioned pool ID so reopened frames cannot collide."""
    row = connection.execute(
        """
        SELECT l.query_version
        FROM physical_queries p
        JOIN logical_queries l USING(logical_query_id)
        WHERE p.physical_query_id = ?
        """,
        (physical_query_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Missing formal query: {physical_query_id}")
    return (
        f"FS_V{int(row['query_version']):03d}_"
        f"{physical_query_id}"
    )


def _register_formal_saturation_pools(
    connection: sqlite3.Connection,
    query_ids: List[str],
) -> None:
    """Map each frozen physical query to a deterministic 10k review pool."""
    protocol = read_json(saturation.SATURATION_PROTOCOL_PATH)
    sample_size = int(
        protocol["sampling"]["maximum_openalex_sample_per_query"]
    )
    namespace = str(protocol["sampling"]["random_seed_namespace"])
    width = int(
        protocol["sampling"]["records_per_stratum_by_role"].get(
            "formal_search_family",
            protocol["sampling"]["records_per_stratum_per_review_round"],
        )
    )
    existing_offset = connection.execute(
        """
        SELECT value FROM metadata
        WHERE key = 'formal_review_rank_offset'
        """
    ).fetchone()
    if existing_offset is None:
        completed_iterations = int(
            connection.execute(
                """
                SELECT COALESCE(MAX(iteration), 0)
                FROM discovery_review_rounds
                """
            ).fetchone()[0]
        )
        connection.execute(
            """
            INSERT INTO metadata(key, value)
            VALUES ('formal_review_rank_offset', ?)
            """,
            (str(completed_iterations * width),),
        )
    for physical_id in query_ids:
        row = connection.execute(
            """
            SELECT p.*, l.family_label, l.search_domain_id, l.query_version
            FROM physical_queries p
            JOIN logical_queries l USING(logical_query_id)
            WHERE p.physical_query_id = ?
            """,
            (physical_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Missing formal query: {physical_id}")
        discovery_id = _formal_discovery_query_id(connection, physical_id)
        seed = saturation._stable_seed(
            namespace + "|FORMAL",
            discovery_id,
        )
        body = {
            "expression": row["expression"],
            "filter_expression": row["filter_expression"],
            "sample_size": sample_size,
            "random_seed": seed,
            "physical_query_id": physical_id,
            "query_version": int(row["query_version"]),
        }
        connection.execute(
            """
            INSERT INTO discovery_queries(
                discovery_query_id, query_role, stratum_label, expression,
                filter_expression, sample_size, random_seed, query_hash,
                status, archive_reason
            ) VALUES (?, 'formal_search_family', ?, ?, ?, ?, ?, ?,
                      'active', '')
            ON CONFLICT(discovery_query_id) DO UPDATE SET
                stratum_label = excluded.stratum_label,
                expression = excluded.expression,
                filter_expression = excluded.filter_expression,
                sample_size = excluded.sample_size,
                random_seed = excluded.random_seed,
                query_hash = excluded.query_hash,
                status = 'active',
                archive_reason = ''
            """,
            (
                discovery_id,
                (
                    f"{row['search_domain_id']}|{row['family_label']}|"
                    f"{physical_id}"
                ),
                row["expression"],
                row["filter_expression"],
                sample_size,
                seed,
                json_hash(body),
            ),
        )
        connection.execute(
            """
            INSERT INTO discovery_query_evidence(
                discovery_query_id, source_ids_json, source_dois_json,
                source_phrases_json, derivation_rule
            ) VALUES (?, ?, '[]', '[]', ?)
            ON CONFLICT(discovery_query_id) DO UPDATE SET
                source_ids_json = excluded.source_ids_json,
                derivation_rule = excluded.derivation_rule
            """,
            (
                discovery_id,
                json.dumps(
                    [row["logical_query_id"], physical_id],
                    ensure_ascii=False,
                ),
                (
                    "deterministic maximum-size review pool from a frozen "
                    "H2/PRESS/recall-validated formal query"
                ),
            ),
        )
    connection.commit()


def _map_formal_saturation_pools(
    connection: sqlite3.Connection,
    query_ids: List[str],
) -> None:
    """Expose pool membership through the formal query audit tables."""
    for physical_id in query_ids:
        discovery_id = _formal_discovery_query_id(connection, physical_id)
        run = connection.execute(
            """
            SELECT * FROM discovery_query_runs
            WHERE discovery_query_id = ?
            """,
            (discovery_id,),
        ).fetchone()
        if run is None or int(run["retrieved_rows"]) == 0:
            continue
        connection.execute(
            """
            DELETE FROM query_hits
            WHERE provider = 'OpenAlex'
              AND physical_query_id = ?
              AND run_role = 'formal'
            """,
            (physical_id,),
        )
        connection.execute(
            """
            INSERT INTO query_hits(
                provider, physical_query_id, run_role, record_key, rank
            )
            SELECT 'OpenAlex', ?, 'formal', record_key, sample_rank
            FROM discovery_hits
            WHERE discovery_query_id = ?
            ORDER BY sample_rank, record_key
            """,
            (physical_id, discovery_id),
        )
        validation = connection.execute(
            """
            SELECT reported_total FROM query_runs
            WHERE provider = 'OpenAlex' AND physical_query_id = ?
              AND run_role = 'search_frame_validation_inventory'
            """,
            (physical_id,),
        ).fetchone()
        reported_total = (
            int(validation["reported_total"])
            if validation is not None
            else int(run["reported_sample_total"] or 0)
        )
        query_hash = connection.execute(
            """
            SELECT query_hash FROM physical_queries
            WHERE physical_query_id = ?
            """,
            (physical_id,),
        ).fetchone()["query_hash"]
        connection.execute(
            """
            INSERT INTO query_runs(
                provider, physical_query_id, run_role, query_hash,
                reported_total, retrieved_rows, unique_hits, pages,
                next_cursor, complete, stopped_reason, error, updated_at
            ) VALUES (
                'OpenAlex', ?, 'formal', ?, ?, ?, ?, ?, '', 1,
                'deterministic_evidence_saturation_pool', '', ?
            )
            ON CONFLICT(provider, physical_query_id, run_role) DO UPDATE SET
                query_hash = excluded.query_hash,
                reported_total = excluded.reported_total,
                retrieved_rows = excluded.retrieved_rows,
                unique_hits = excluded.unique_hits,
                pages = excluded.pages,
                next_cursor = '',
                complete = 1,
                stopped_reason = excluded.stopped_reason,
                error = '',
                updated_at = excluded.updated_at
            """,
            (
                physical_id,
                query_hash,
                reported_total,
                int(run["retrieved_rows"]),
                int(run["unique_hits"]),
                int(run["pages"]),
                utc_now(),
            ),
        )
    connection.commit()


def mark_formal_retrieval_stage(
    connection: sqlite3.Connection,
) -> Dict[str, Any]:
    """Reconcile formal completion from all active checkpoints."""
    query_ids = formal_query_ids(connection)
    if not query_ids:
        raise RuntimeError("No active formal physical queries")
    rows = connection.execute(
        f"""
        SELECT * FROM query_runs
        WHERE provider = 'OpenAlex' AND run_role = 'formal'
          AND physical_query_id IN ({','.join('?' for _ in query_ids)})
        ORDER BY physical_query_id
        """,
        query_ids,
    ).fetchall()
    complete = len(rows) == len(query_ids) and all(row["complete"] for row in rows)
    details = {
        "P": len(query_ids),
        "retrieval_design": "deterministic_per_query_saturation_pools",
        "broad_query_results_exhaustive": False,
        "complete_queries": sum(int(row["complete"]) for row in rows),
        "reported_total_sum": sum(
            int(row["reported_total"] or 0) for row in rows
        ),
        "retrieved_rows": sum(int(row["retrieved_rows"]) for row in rows),
        "unique_query_record_links": connection.execute(
            """
            SELECT COUNT(*) FROM query_hits
            WHERE provider = 'OpenAlex' AND run_role = 'formal'
            """
        ).fetchone()[0],
        "unique_records": connection.execute(
            """
            SELECT COUNT(DISTINCT record_key) FROM query_hits
            WHERE provider = 'OpenAlex' AND run_role = 'formal'
            """
        ).fetchone()[0],
    }
    set_stage(
        connection,
        "formal_retrieval_complete",
        "complete" if complete else "ready",
        details,
    )
    connection.commit()
    return details


def _citation_sources(
    connection: sqlite3.Connection,
    scope: str,
) -> List[sqlite3.Row]:
    if scope == "included":
        keys = included_record_keys(connection)
    elif scope == "indicator_sources":
        keys = [
            str(row[0])
            for row in connection.execute(
                """
                SELECT d.record_key
                FROM indicator_source_disposition d
                JOIN indicator_source_reviews h2
                  ON h2.record_key = d.record_key
                 AND h2.reviewer_role = 'H2'
                WHERE d.disposition IN (
                    'extracted', 'candidate_fulltext_missing'
                )
                ORDER BY d.record_key
                """
            )
        ]
    elif scope == "reviews_and_indicator_sources":
        keys = [
            str(row[0])
            for row in connection.execute(
                """
                SELECT r.record_key
                FROM records r
                JOIN screening_final s USING(record_key)
                WHERE s.final_decision = 'include'
                  AND s.final_language = 'en'
                  AND r.work_type = 'review'
                UNION
                SELECT d.record_key
                FROM indicator_source_disposition d
                JOIN indicator_source_reviews h2
                  ON h2.record_key = d.record_key
                 AND h2.reviewer_role = 'H2'
                WHERE d.disposition IN (
                    'extracted', 'candidate_fulltext_missing'
                )
                ORDER BY record_key
                """
            )
        ]
    else:
        raise ValueError(
            "Citation scope must be included, indicator_sources, or "
            "reviews_and_indicator_sources"
        )
    if not keys:
        return []
    return connection.execute(
        f"""
        SELECT * FROM records
        WHERE provider = 'OpenAlex'
          AND record_key IN ({','.join('?' for _ in keys)})
        ORDER BY record_key
        """,
        keys,
    ).fetchall()


def _register_citation_review_network(
    connection: sqlite3.Connection,
    iteration: int,
    sources: List[sqlite3.Row],
    scope: str,
) -> int:
    """Add citation-tracked records to deterministic formal review rounds."""
    scope_id = re.sub(r"[^A-Z0-9]+", "_", scope.upper()).strip("_")
    query_id = f"CS_{scope_id}_ITER_{iteration:03d}"
    record_keys = [
        str(row["record_key"])
        for row in connection.execute(
            """
            SELECT DISTINCT r.record_key
            FROM citation_edges e
            JOIN records r ON r.provider_id = e.target_provider_id
            WHERE e.iteration = ?
              AND e.eligibility_status = 'pending_screening'
            ORDER BY r.record_key
            """,
            (iteration,),
        )
    ]
    reviewed_existing = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM discovery_hits
            WHERE discovery_query_id = ? AND review_round > 0
            """,
            (query_id,),
        ).fetchone()[0]
    )
    existing_keys = {
        str(row[0])
        for row in connection.execute(
            """
            SELECT record_key FROM discovery_hits
            WHERE discovery_query_id = ?
            """,
            (query_id,),
        )
    }
    if reviewed_existing and existing_keys != set(record_keys):
        raise RuntimeError(
            "A reviewed citation-network iteration is immutable; use a new "
            "iteration for newly discovered records"
        )
    body = {
        "iteration": iteration,
        "source_record_keys": [
            str(source["record_key"]) for source in sources
        ],
        "record_keys": record_keys,
    }
    connection.execute(
        """
        INSERT INTO discovery_queries(
            discovery_query_id, query_role, stratum_label, expression,
            filter_expression, sample_size, random_seed, query_hash,
            status, archive_reason
        ) VALUES (?, 'citation_tracking_network', ?, '', '', ?, 0, ?,
                  'network', '')
        ON CONFLICT(discovery_query_id) DO UPDATE SET
            sample_size = excluded.sample_size,
            query_hash = excluded.query_hash,
            status = 'network',
            archive_reason = ''
        """,
        (
            query_id,
            f"formal citation tracking iteration {iteration}",
            len(record_keys),
            json_hash(body),
        ),
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
            json.dumps(
                [str(source["record_key"]) for source in sources],
                ensure_ascii=False,
            ),
            json.dumps(
                sorted(
                    {
                        str(source["doi"])
                        for source in sources
                        if str(source["doi"])
                    }
                ),
                ensure_ascii=False,
            ),
            (
                "complete eligible forward/backward citation records from "
                "included review or indicator evidence sources"
            ),
        ),
    )
    saturation_protocol = read_json(saturation.SATURATION_PROTOCOL_PATH)
    citation_width = int(
        saturation_protocol["sampling"]["records_per_stratum_by_role"].get(
            "citation_tracking_network",
            saturation_protocol["sampling"][
                "records_per_stratum_per_review_round"
            ],
        )
    )
    rank_offset = int(
        connection.execute(
            """
            SELECT COALESCE(MAX(iteration), 0)
            FROM discovery_review_rounds
            """
        ).fetchone()[0]
    ) * citation_width
    for rank, record_key in enumerate(record_keys, start=1):
        connection.execute(
            """
            INSERT OR IGNORE INTO discovery_hits(
                discovery_query_id, record_key, sample_rank,
                selection_hash, review_rank, review_round, review_status
            ) VALUES (?, ?, ?, ?, ?, 0, 'unassigned')
            """,
            (
                query_id,
                record_key,
                rank,
                saturation._selection_hash(query_id, record_key),
                rank_offset + rank,
            ),
        )
    connection.commit()
    return len(record_keys)


def _compact_openalex_id(value: Any) -> str:
    """Return a canonical short OpenAlex work identity."""
    short_id = str(value or "").strip().rstrip("/").rsplit("/", maxsplit=1)[-1]
    return short_id if re.fullmatch(r"W[0-9]+", short_id) else ""


def _chunks(values: Sequence[str], size: int) -> List[List[str]]:
    if size < 1:
        raise ValueError("Citation batch size must be positive")
    return [
        list(values[start : start + size])
        for start in range(0, len(values), size)
    ]


def _known_citation_targets(
    connection: sqlite3.Connection,
    provider_ids: Sequence[str],
    cutoff_year: int,
) -> tuple[set[str], set[str]]:
    """Return locally known and locally eligible OpenAlex work IDs."""
    known: set[str] = set()
    eligible: set[str] = set()
    for chunk in _chunks(list(provider_ids), 500):
        if not chunk:
            continue
        full_ids = [f"https://openalex.org/{value}" for value in chunk]
        placeholders = ",".join("?" for _ in full_ids)
        for row in connection.execute(
            f"""
            SELECT provider_id, publication_year, work_type
            FROM records
            WHERE provider = 'OpenAlex'
              AND provider_id IN ({placeholders})
            """,
            full_ids,
        ):
            short_id = _compact_openalex_id(row["provider_id"])
            if not short_id:
                continue
            known.add(short_id)
            year = row["publication_year"]
            if (
                row["work_type"] in {"article", "review"}
                and (year is None or int(year) <= cutoff_year)
            ):
                eligible.add(short_id)
    return known, eligible


def _citation_filter(
    direction: str,
    identities: Sequence[str],
    cutoff_date: str,
) -> str:
    field = "ids.openalex" if direction == "backward" else "cites"
    return (
        f"{field}:{'|'.join(identities)},"
        f"to_publication_date:{cutoff_date},type:article|review"
    )


def _register_citation_queries(
    connection: sqlite3.Connection,
    iteration: int,
    direction: str,
    scope_identities: Sequence[str],
    request_identities: Sequence[str],
    batch_size: int,
    cutoff_date: str,
) -> List[str]:
    """Freeze resumable provider batches for one citation direction."""
    logical_id = f"CIT{iteration:03d}_{direction.upper()}"
    scope_values = sorted(set(scope_identities))
    request_values = sorted(set(request_identities))
    existing = connection.execute(
        """
        SELECT object_terms_json FROM logical_queries
        WHERE logical_query_id = ?
        """,
        (logical_id,),
    ).fetchone()
    if existing is not None:
        if json.loads(existing["object_terms_json"]) != scope_values:
            raise RuntimeError(
                "A citation iteration has a different frozen source scope; "
                "use a new iteration"
            )
        return [
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
    definition = {
        "iteration": iteration,
        "direction": direction,
        "scope_identities": scope_values,
        "request_identities": request_values,
        "batch_size": batch_size,
        "cutoff_date": cutoff_date,
    }
    connection.execute(
        """
        INSERT INTO logical_queries(
            logical_query_id, query_version, search_domain_id,
            family_label, logical_expression, object_terms_json,
            domain_terms_json, context_terms_json, status,
            archive_reason, press_status, press_reviewer, press_notes,
            query_hash
        ) VALUES (?, 1, 'CITATION_TRACKING', ?, '', ?, ?, ?, 'citation',
                  '', 'not_applicable', 'SYSTEM',
                  'Exact source-derived citation filter', ?)
        """,
        (
            logical_id,
            f"citation_tracking_{direction}_iteration_{iteration}",
            json.dumps(scope_values, ensure_ascii=False),
            json.dumps(request_values, ensure_ascii=False),
            json.dumps(
                {"cutoff_date": cutoff_date, "batch_size": batch_size},
                ensure_ascii=False,
                sort_keys=True,
            ),
            json_hash(definition),
        ),
    )
    physical_ids: List[str] = []
    for index, chunk in enumerate(
        _chunks(request_values, batch_size),
        start=1,
    ):
        physical_id = f"{logical_id}__P{index:03d}"
        filter_expression = _citation_filter(
            direction,
            chunk,
            cutoff_date,
        )
        connection.execute(
            """
            INSERT INTO physical_queries(
                physical_query_id, logical_query_id, provider, expression,
                filter_expression, status, query_hash
            ) VALUES (?, ?, 'OpenAlex', '', ?, 'active', ?)
            """,
            (
                physical_id,
                logical_id,
                filter_expression,
                query_definition_hash("", filter_expression),
            ),
        )
        physical_ids.append(physical_id)
    connection.commit()
    return physical_ids


def _budgeted_citation_fetcher(
    connection: sqlite3.Connection,
) -> Any:
    """Adapt resumable query URLs to the free-credit-aware scheduler."""
    scheduler = saturation.OpenAlexBudgetScheduler(connection)

    def fetch(url: str) -> Dict[str, Any]:
        parsed = urllib.parse.urlparse(url)
        values = urllib.parse.parse_qs(
            parsed.query,
            keep_blank_values=True,
        )
        parameters: Dict[str, Any] = {
            key: entries[-1]
            for key, entries in values.items()
            if entries and key != "api_key"
        }
        return scheduler.fetch_search(parameters)

    return fetch


def _run_citation_queries(
    connection: sqlite3.Connection,
    physical_ids: Sequence[str],
    iteration: int,
    fetcher: Any | None,
    key_offset: int,
) -> List[Dict[str, Any]]:
    """Execute all batches and require complete cursor checkpoints."""
    runs: List[Dict[str, Any]] = []
    for index, physical_id in enumerate(physical_ids):
        kwargs: Dict[str, Any] = {
            "api_key_offset": key_offset + index,
        }
        if fetcher is not None:
            kwargs["fetcher"] = fetcher
        run = retrieve_physical_query(
            connection,
            physical_id,
            f"citation_tracking_{iteration:03d}",
            **kwargs,
        )
        if not bool(run["complete"]):
            raise RuntimeError(
                f"Incomplete citation query checkpoint: {physical_id}"
            )
        runs.append(run)
    return runs


def _citation_hit_records(
    connection: sqlite3.Connection,
    physical_ids: Sequence[str],
    iteration: int,
) -> List[sqlite3.Row]:
    if not physical_ids:
        return []
    placeholders = ",".join("?" for _ in physical_ids)
    return connection.execute(
        f"""
        SELECT DISTINCT r.record_key, r.provider_id,
               r.referenced_works_json
        FROM query_hits h
        JOIN records r
          ON r.provider = h.provider AND r.record_key = h.record_key
        WHERE h.run_role = ?
          AND h.physical_query_id IN ({placeholders})
        ORDER BY r.record_key
        """,
        [f"citation_tracking_{iteration:03d}", *physical_ids],
    ).fetchall()


def _upsert_citation_edge(
    connection: sqlite3.Connection,
    source_record_key: str,
    target_short_id: str,
    direction: str,
    iteration: int,
    status: str,
) -> None:
    connection.execute(
        """
        INSERT INTO citation_edges(
            source_record_key, target_provider_id, direction,
            iteration, eligibility_status
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(
            source_record_key, target_provider_id, direction, iteration
        ) DO UPDATE SET eligibility_status = excluded.eligibility_status
        """,
        (
            source_record_key,
            f"https://openalex.org/{target_short_id}",
            direction,
            iteration,
            status,
        ),
    )


def track_citations(
    connection: sqlite3.Connection,
    iteration: int,
    scope: str = "reviews_and_indicator_sources",
    fetcher: Any | None = None,
) -> Dict[str, Any]:
    """Retrieve forward/backward citations for reviewed English sources."""
    if iteration < 1:
        raise ValueError("Citation iteration must be at least 1")
    require_complete(connection, ["literature_screened"])
    protocol = read_json(PROTOCOL_PATH)
    sources = _citation_sources(connection, scope)
    if not sources:
        raise RuntimeError(f"No citation sources are available for {scope}")
    before = connection.execute(
        "SELECT COUNT(*) FROM records"
    ).fetchone()[0]
    cutoff_date = str(protocol["cutoff_date"])
    api_protocol = read_json(saturation.SATURATION_PROTOCOL_PATH)["api"]
    backward_size = int(api_protocol["citation_backward_id_batch_size"])
    forward_size = int(api_protocol["citation_forward_source_batch_size"])
    citation_fetcher = (
        fetcher
        if fetcher is not None
        else _budgeted_citation_fetcher(connection)
    )
    source_references: Dict[str, set[str]] = {}
    for source in sources:
        raw_references = json.loads(source["referenced_works_json"])
        if not isinstance(raw_references, list):
            raw_references = []
        source_references[str(source["record_key"])] = {
            value
            for value in (
                _compact_openalex_id(reference)
                for reference in raw_references
            )
            if value
        }
    backward_scope = sorted(
        {
            identity
            for values in source_references.values()
            for identity in values
        }
    )
    known_backward, locally_eligible = _known_citation_targets(
        connection,
        backward_scope,
        int(cutoff_date[:4]),
    )
    backward_requests = sorted(set(backward_scope) - known_backward)
    backward_query_ids = _register_citation_queries(
        connection,
        iteration,
        "backward",
        backward_scope,
        backward_requests,
        backward_size,
        cutoff_date,
    )
    backward_runs = _run_citation_queries(
        connection,
        backward_query_ids,
        iteration,
        citation_fetcher,
        0,
    )
    backward_records = _citation_hit_records(
        connection,
        backward_query_ids,
        iteration,
    )
    api_eligible = {
        _compact_openalex_id(row["provider_id"])
        for row in backward_records
    }
    eligible_backward = locally_eligible | {
        value for value in api_eligible if value
    }
    for source_key, references in source_references.items():
        for target_id in sorted(references):
            _upsert_citation_edge(
                connection,
                source_key,
                target_id,
                "backward",
                iteration,
                (
                    "pending_screening"
                    if target_id in eligible_backward
                    else "out_of_scope_or_unindexed"
                ),
            )
    source_by_id = {
        _compact_openalex_id(source["provider_id"]): str(
            source["record_key"]
        )
        for source in sources
        if _compact_openalex_id(source["provider_id"])
    }
    forward_scope = sorted(source_by_id)
    forward_query_ids = _register_citation_queries(
        connection,
        iteration,
        "forward",
        forward_scope,
        forward_scope,
        forward_size,
        cutoff_date,
    )
    forward_runs = _run_citation_queries(
        connection,
        forward_query_ids,
        iteration,
        citation_fetcher,
        len(backward_query_ids),
    )
    forward_records = _citation_hit_records(
        connection,
        forward_query_ids,
        iteration,
    )
    for record in forward_records:
        references = {
            value
            for value in (
                _compact_openalex_id(reference)
                for reference in json.loads(record["referenced_works_json"])
            )
            if value
        }
        matched_sources = sorted(references.intersection(source_by_id))
        if not matched_sources:
            raise RuntimeError(
                "OpenAlex returned a forward-citation record without a "
                "matching referenced-work identity"
            )
        target_id = _compact_openalex_id(record["provider_id"])
        for source_id in matched_sources:
            _upsert_citation_edge(
                connection,
                source_by_id[source_id],
                target_id,
                "forward",
                iteration,
                "pending_screening",
            )
    connection.commit()
    after = connection.execute(
        "SELECT COUNT(*) FROM records"
    ).fetchone()[0]
    new_records = after - before
    citation_review_records = _register_citation_review_network(
        connection,
        iteration,
        sources,
        scope,
    )
    connection.execute(
        """
        INSERT INTO saturation_rounds(
            iteration, new_records,
            new_nonredundant_english_terms,
            new_canonical_indicator_families, reviewer_role,
            decision, notes, reviewed_at
        ) VALUES (?, ?, -1, -1, 'SYSTEM', 'pending',
                  'Awaiting H2 saturation coding', ?)
        ON CONFLICT(iteration) DO UPDATE SET
            new_records = excluded.new_records,
            new_nonredundant_english_terms = -1,
            new_canonical_indicator_families = -1,
            reviewer_role = 'SYSTEM',
            decision = 'pending',
            notes = 'Awaiting H2 saturation coding',
            reviewed_at = excluded.reviewed_at
        """,
        (iteration, new_records, utc_now()),
    )
    details = {
        "iteration": iteration,
        "scope": scope,
        "citation_sources": len(sources),
        "backward_edges_seen": sum(
            len(values) for values in source_references.values()
        ),
        "backward_records_in_scope": len(eligible_backward),
        "backward_records_reused_locally": len(known_backward),
        "backward_ids_unresolved_or_out_of_scope": len(
            set(backward_scope) - eligible_backward
        ),
        "forward_records_seen": len(forward_records),
        "citation_physical_queries": (
            len(backward_query_ids) + len(forward_query_ids)
        ),
        "citation_api_pages": sum(
            int(run["pages"]) for run in [*backward_runs, *forward_runs]
        ),
        "api_key_rotation": (
            "free_credit_scheduler_round_robin"
            if fetcher is None
            else "caller_supplied_fetcher"
        ),
        "new_unique_records": new_records,
        "citation_review_network_records": citation_review_records,
    }
    if new_records:
        set_stage(
            connection,
            "literature_screened",
            "ready",
            {
                "reason": "new citation-tracked records require screening",
                **details,
            },
        )
        for stage in (
            "indicators_extracted",
            "dimensions_derived",
            "features_selected",
            "audit_complete",
        ):
            set_stage(
                connection,
                stage,
                "pending",
                {"reason": f"citation tracking iteration {iteration}"},
            )
    log_event(
        connection,
        "citation_tracking",
        "iteration",
        str(iteration),
        details,
    )
    connection.commit()
    return details


def record_saturation_round(
    connection: sqlite3.Connection,
    iteration: int,
    new_records: int,
    new_terms: int,
    new_indicator_families: int,
    decision: str,
    notes: str,
    reviewer_role: str = "H2",
) -> Dict[str, Any]:
    """Record H2's saturation decision after term/indicator reconciliation."""
    if any(value < 0 for value in (new_records, new_terms, new_indicator_families)):
        raise ValueError("Saturation counts cannot be negative")
    reviewer_role = reviewer_role.upper()
    decision = decision.casefold()
    if reviewer_role != "H2":
        raise ValueError("Only H2 can approve a saturation round")
    if decision not in {"continue", "freeze"}:
        raise ValueError("Saturation decision must be continue or freeze")
    if decision == "freeze" and any(
        value != 0 for value in (new_terms, new_indicator_families)
    ):
        raise ValueError(
            "A freeze round requires zero new non-redundant English terms "
            "and zero new canonical indicator families"
        )
    if not notes.strip():
        raise ValueError("Saturation decisions require notes")
    existing = connection.execute(
        "SELECT new_records FROM saturation_rounds WHERE iteration = ?",
        (iteration,),
    ).fetchone()
    if existing is not None and int(existing["new_records"]) != new_records:
        raise ValueError(
            "new_records does not match the citation retrieval checkpoint"
        )
    connection.execute(
        """
        INSERT INTO saturation_rounds(
            iteration, new_records, new_nonredundant_english_terms,
            new_canonical_indicator_families, reviewer_role, decision,
            notes, reviewed_at
        ) VALUES (?, ?, ?, ?, 'H2', ?, ?, ?)
        ON CONFLICT(iteration) DO UPDATE SET
            new_nonredundant_english_terms =
                excluded.new_nonredundant_english_terms,
            new_canonical_indicator_families =
                excluded.new_canonical_indicator_families,
            reviewer_role = 'H2',
            decision = excluded.decision,
            notes = excluded.notes,
            reviewed_at = excluded.reviewed_at
        """,
        (
            iteration,
            new_records,
            new_terms,
            new_indicator_families,
            decision,
            notes.strip(),
            utc_now(),
        ),
    )
    payload = {
        "iteration": iteration,
        "new_records": new_records,
        "new_nonredundant_english_terms": new_terms,
        "new_canonical_indicator_families": new_indicator_families,
        "decision": decision,
    }
    log_event(
        connection,
        "saturation_review",
        "iteration",
        str(iteration),
        payload,
    )
    set_stage(
        connection,
        "audit_complete",
        "pending",
        {"reason": f"saturation decision changed at iteration {iteration}"},
    )
    connection.commit()
    return payload
