from __future__ import annotations

import json
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from common import (
    json_hash,
    local_environment_value,
    normalize_doi,
    normalize_text,
    read_csv,
    sha256_bytes,
    utc_now,
    write_csv,
)
from database import invalidate_stages, log_event, snapshot_import_file


OPENALEX_BASE_URL = "https://api.openalex.org"
CROSSREF_BASE_URL = "https://api.crossref.org"
USER_AGENT = (
    "ASPR-evidence-derived-v3/3.0 "
    "(systematic evidence-saturation map; contact via repository owner)"
)


def safe_provider_error(error: Exception) -> str:
    """Return an error summary with URLs and API-key values removed."""
    if isinstance(error, urllib.error.HTTPError):
        return f"HTTPError status={error.code}"
    if isinstance(error, urllib.error.URLError):
        return f"URLError reason={type(error.reason).__name__}"
    message = str(error)
    message = re.sub(
        r"(?i)(api_key=)[^&\s]+",
        r"\1[REDACTED]",
        message,
    )
    message = re.sub(r"https?://\S+", "[URL_REDACTED]", message)
    return f"{type(error).__name__}: {message}"[:1000]


def fetch_json(
    url: str,
    retries: int = 6,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    """Fetch a JSON object with bounded exponential retry."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_seconds,
            ) as response:
                value = json.loads(response.read().decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Provider response is not a JSON object")
            return value
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            last_error = error
            if isinstance(error, urllib.error.HTTPError) and error.code in {
                400,
                401,
                403,
                404,
                410,
                422,
            }:
                break
            if attempt + 1 >= retries:
                break
            retry_after = 0.0
            if isinstance(error, urllib.error.HTTPError):
                try:
                    retry_after = float(
                        error.headers.get("Retry-After", "0")
                    )
                except ValueError:
                    retry_after = 0.0
            time.sleep(min(max(retry_after, float(2**attempt)), 45.0))
    if last_error is None:
        raise RuntimeError("Provider request failed without an error object")
    raise RuntimeError(
        "Provider request failed after retries: "
        + safe_provider_error(last_error)
    )


def openalex_api_keys() -> List[str]:
    """Return configured OpenAlex keys without logging or persisting them."""
    raw_values = [
        local_environment_value("OPENALEX_API_KEYS"),
        local_environment_value("OPENALEX_API_KEY"),
    ]
    keys: List[str] = []
    for raw_value in raw_values:
        for candidate in raw_value.replace(";", ",").split(","):
            key = candidate.strip()
            if key and key not in keys:
                keys.append(key)
    return keys


def _openalex_parameters(
    *,
    expression: str = "",
    filter_expression: str = "",
    cursor: str = "*",
    per_page: int = 100,
    api_key: str = "",
    select: str = "",
) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {
        "per_page": per_page,
        "cursor": cursor,
    }
    if expression:
        parameters["search"] = expression
    if filter_expression:
        parameters["filter"] = filter_expression
    if select:
        parameters["select"] = select
    if api_key:
        parameters["api_key"] = api_key
    return parameters


def openalex_url(
    *,
    expression: str = "",
    filter_expression: str = "",
    cursor: str = "*",
    per_page: int = 100,
    api_key: str = "",
    select: str = "",
) -> str:
    """Build one OpenAlex works request URL."""
    parameters = _openalex_parameters(
        expression=expression,
        filter_expression=filter_expression,
        cursor=cursor,
        per_page=per_page,
        api_key=api_key,
        select=select,
    )
    return (
        f"{OPENALEX_BASE_URL}/works?"
        + urllib.parse.urlencode(parameters)
    )


def reconstruct_abstract(inverted_index: Any) -> str:
    """Reconstruct an OpenAlex abstract from an inverted index."""
    if not isinstance(inverted_index, dict):
        return ""
    positioned: List[tuple[int, str]] = []
    for token, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                positioned.append((position, str(token)))
    return " ".join(token for _, token in sorted(positioned))


def make_record_key(
    doi: str,
    provider_id: str,
    title: str,
    publication_year: int | None,
) -> str:
    """Apply DOI, provider-ID, then normalized-title/year identity."""
    normalized_doi = normalize_doi(doi)
    if normalized_doi:
        return f"doi:{normalized_doi}"
    if provider_id:
        compact_id = provider_id.rsplit("/", maxsplit=1)[-1].casefold()
        return f"openalex:{compact_id}"
    identity = f"{normalize_text(title)}|{publication_year or ''}"
    return "title-year:" + sha256_bytes(identity.encode("utf-8"))[:24]


def openalex_record(
    item: Mapping[str, Any],
    retrieval_route: str,
) -> Dict[str, Any]:
    """Normalize one OpenAlex work for storage."""
    provider_id = str(item.get("id") or "")
    doi = normalize_doi(item.get("doi"))
    title = str(item.get("display_name") or item.get("title") or "").strip()
    raw_year = item.get("publication_year")
    publication_year = raw_year if isinstance(raw_year, int) else None
    primary_location = item.get("primary_location")
    source_url = ""
    if isinstance(primary_location, dict):
        source_url = str(
            primary_location.get("landing_page_url")
            or primary_location.get("pdf_url")
            or ""
        )
    if not source_url:
        source_url = provider_id
    references = item.get("referenced_works")
    if not isinstance(references, list):
        references = []
    return {
        "provider": "OpenAlex",
        "record_key": make_record_key(
            doi,
            provider_id,
            title,
            publication_year,
        ),
        "provider_id": provider_id,
        "doi": doi,
        "title": title,
        "abstract": reconstruct_abstract(
            item.get("abstract_inverted_index")
        ),
        "language": str(item.get("language") or "unknown").casefold(),
        "publication_year": publication_year,
        "work_type": str(item.get("type") or ""),
        "source_url": source_url,
        "referenced_works_json": json.dumps(
            references,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "raw_json": json.dumps(
            dict(item),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "retrieval_route": retrieval_route,
        "first_seen_at": utc_now(),
    }


def insert_openalex_record(
    connection: sqlite3.Connection,
    record: Mapping[str, Any],
) -> None:
    """Insert a work without overwriting its first provenance."""
    def payload_digest(value: Mapping[str, Any]) -> str:
        payload = "\0".join(
            (
                str(value.get("abstract") or ""),
                str(value.get("referenced_works_json") or "[]"),
                str(value.get("raw_json") or ""),
            )
        )
        return sha256_bytes(payload.encode("utf-8"))

    def save_digest(value: Mapping[str, Any]) -> None:
        connection.execute(
            """
            INSERT INTO record_payload_digests(
                provider, record_key, payload_sha256
            ) VALUES (?, ?, ?)
            ON CONFLICT(provider, record_key) DO UPDATE SET
                payload_sha256 = excluded.payload_sha256
            """,
            (
                str(value["provider"]),
                str(value["record_key"]),
                payload_digest(value),
            ),
        )

    existing = connection.execute(
        "SELECT provider FROM records WHERE record_key = ?",
        (record["record_key"],),
    ).fetchone()
    if existing is not None and existing["provider"] != "OpenAlex":
        connection.execute(
            """
            DELETE FROM record_payload_digests
            WHERE provider = ? AND record_key = ?
            """,
            (existing["provider"], record["record_key"]),
        )
        connection.execute(
            """
            UPDATE records
            SET provider = 'OpenAlex', provider_id = :provider_id,
                doi = :doi, title = :title, abstract = :abstract,
                language = :language, publication_year = :publication_year,
                work_type = :work_type, source_url = :source_url,
                referenced_works_json = :referenced_works_json,
                raw_json = :raw_json,
                retrieval_route = retrieval_route || '|' || :retrieval_route
            WHERE record_key = :record_key
            """,
            dict(record),
        )
        save_digest({**dict(record), "provider": "OpenAlex"})
        return
    cursor = connection.execute(
        """
        INSERT OR IGNORE INTO records(
            provider, record_key, provider_id, doi, title, abstract,
            language, publication_year, work_type, source_url,
            referenced_works_json, raw_json, retrieval_route, first_seen_at
        ) VALUES (
            :provider, :record_key, :provider_id, :doi, :title, :abstract,
            :language, :publication_year, :work_type, :source_url,
            :referenced_works_json, :raw_json, :retrieval_route,
            :first_seen_at
        )
        """,
        dict(record),
    )
    if cursor.rowcount:
        save_digest(record)
        return
    digest_exists = connection.execute(
        """
        SELECT 1 FROM record_payload_digests
        WHERE provider = ? AND record_key = ?
        """,
        (record["provider"], record["record_key"]),
    ).fetchone()
    if digest_exists is None:
        stored = connection.execute(
            """
            SELECT provider, record_key, abstract, referenced_works_json,
                   raw_json
            FROM records
            WHERE provider = ? AND record_key = ?
            """,
            (record["provider"], record["record_key"]),
        ).fetchone()
        if stored is not None:
            save_digest(stored)


def query_inventory(
    expression: str,
    filter_expression: str,
    api_key: str = "",
    fetcher: Any = fetch_json,
) -> int:
    """Return the provider-reported count without paging results."""
    url = openalex_url(
        expression=expression,
        filter_expression=filter_expression,
        per_page=1,
        api_key=api_key,
        select="id",
    )
    payload = fetcher(url)
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("OpenAlex response has no meta object")
    count = meta.get("count")
    if not isinstance(count, int):
        raise ValueError("OpenAlex meta.count is not an integer")
    return count


def inventory_physical_queries(
    connection: sqlite3.Connection,
    query_ids: Sequence[str],
    run_role: str,
    fetcher: Any = fetch_json,
) -> Dict[str, int]:
    """Inventory physical queries and persist zero-hit evidence."""
    keys = openalex_api_keys()
    totals: Dict[str, int] = {}
    for index, query_id in enumerate(query_ids):
        row = connection.execute(
            """
            SELECT expression, filter_expression, query_hash
            FROM physical_queries
            WHERE physical_query_id = ? AND status = 'active'
            """,
            (query_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown active physical query: {query_id}")
        api_key = keys[index % len(keys)] if keys else ""
        total = query_inventory(
            str(row["expression"]),
            str(row["filter_expression"]),
            api_key,
            fetcher,
        )
        totals[query_id] = total
        connection.execute(
            """
            INSERT INTO query_runs(
                provider, physical_query_id, run_role, query_hash,
                reported_total, retrieved_rows, unique_hits, pages,
                next_cursor, complete, stopped_reason, error, updated_at
            ) VALUES (
                'OpenAlex', ?, ?, ?, ?, 0, 0, 0, '*', 1,
                'inventory_only', '', ?
            )
            ON CONFLICT(provider, physical_query_id, run_role) DO UPDATE SET
                query_hash = excluded.query_hash,
                reported_total = excluded.reported_total,
                complete = 1,
                stopped_reason = 'inventory_only',
                error = '',
                updated_at = excluded.updated_at
            """,
            (query_id, run_role, row["query_hash"], total, utc_now()),
        )
        connection.commit()
    return totals


def _retrieval_select() -> str:
    return (
        "id,doi,display_name,publication_year,type,language,"
        "abstract_inverted_index,primary_location,best_oa_location,"
        "referenced_works"
    )


def hydrate_openalex_locations(
    connection: sqlite3.Connection,
    record_keys: Iterable[str],
    fetcher: Any = fetch_json,
    retry_failed: bool = False,
) -> Dict[str, int]:
    """Refresh OA locations only for the small finally included source set."""
    keys = openalex_api_keys()
    selected_keys = sorted(set(record_keys))
    counts = {
        "records": len(selected_keys),
        "hydrated": 0,
        "resumed": 0,
        "failed": 0,
        "unconfigured": 0,
    }
    if not keys:
        counts["unconfigured"] = len(selected_keys)
        return counts
    for index, record_key in enumerate(selected_keys):
        existing = connection.execute(
            """
            SELECT status FROM openalex_location_hydration
            WHERE record_key = ?
            """,
            (record_key,),
        ).fetchone()
        if existing is not None and (
            existing["status"] == "complete"
            or (existing["status"] == "error" and not retry_failed)
        ):
            counts["resumed"] += 1
            continue
        record = connection.execute(
            """
            SELECT provider_id, raw_json FROM records
            WHERE record_key = ? AND provider = 'OpenAlex'
            """,
            (record_key,),
        ).fetchone()
        if record is None:
            continue
        provider_id = str(record["provider_id"] or "")
        short_id = provider_id.rstrip("/").rsplit("/", maxsplit=1)[-1]
        if not short_id.startswith("W"):
            continue
        payload: Dict[str, Any] | None = None
        last_error: Exception | None = None
        for offset in range(len(keys)):
            slot = (index + offset) % len(keys)
            parameters = urllib.parse.urlencode(
                {
                    "select": (
                        "id,primary_location,best_oa_location,locations"
                    ),
                    "api_key": keys[slot],
                }
            )
            try:
                value = fetcher(
                    f"{OPENALEX_BASE_URL}/works/{short_id}?{parameters}"
                )
                if not isinstance(value, dict):
                    raise ValueError("Malformed OpenAlex location response")
                payload = value
                break
            except Exception as error:
                last_error = error
        if payload is None:
            connection.execute(
                """
                INSERT INTO openalex_location_hydration(
                    record_key, status, provider_payload_sha256,
                    error, hydrated_at
                ) VALUES (?, 'error', '', ?, ?)
                ON CONFLICT(record_key) DO UPDATE SET
                    status = 'error',
                    provider_payload_sha256 = '',
                    error = excluded.error,
                    hydrated_at = excluded.hydrated_at
                """,
                (
                    record_key,
                    safe_provider_error(
                        last_error or RuntimeError("unknown")
                    ),
                    utc_now(),
                ),
            )
            counts["failed"] += 1
            connection.commit()
            continue
        raw = json.loads(str(record["raw_json"] or "{}"))
        for field in (
            "primary_location",
            "best_oa_location",
            "locations",
        ):
            if field in payload:
                raw[field] = payload[field]
        provider_payload_hash = json_hash(
            {
                field: payload.get(field)
                for field in (
                    "id",
                    "primary_location",
                    "best_oa_location",
                    "locations",
                )
            }
        )
        connection.execute(
            "UPDATE records SET raw_json = ? WHERE record_key = ?",
            (
                json.dumps(
                    raw,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                record_key,
            ),
        )
        connection.execute(
            """
            INSERT INTO openalex_location_hydration(
                record_key, status, provider_payload_sha256,
                error, hydrated_at
            ) VALUES (?, 'complete', ?, '', ?)
            ON CONFLICT(record_key) DO UPDATE SET
                status = 'complete',
                provider_payload_sha256 =
                    excluded.provider_payload_sha256,
                error = '',
                hydrated_at = excluded.hydrated_at
            """,
            (record_key, provider_payload_hash, utc_now()),
        )
        counts["hydrated"] += 1
        connection.commit()
    log_event(
        connection,
        "openalex_location_hydration",
        "scope",
        "included_english_sources",
        {
            **counts,
            "configured_key_slots": len(keys),
        },
    )
    invalidate_stages(
        connection,
        ("audit_complete",),
        "OpenAlex location evidence changed",
    )
    connection.commit()
    return counts


def retrieve_physical_query(
    connection: sqlite3.Connection,
    physical_query_id: str,
    run_role: str,
    per_page: int = 100,
    max_pages: int | None = None,
    fetcher: Any = fetch_json,
    api_key_offset: int = 0,
) -> Dict[str, Any]:
    """Retrieve one physical query with a committed cursor checkpoint."""
    row = connection.execute(
        """
        SELECT expression, filter_expression, query_hash, status
        FROM physical_queries
        WHERE physical_query_id = ?
        """,
        (physical_query_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown physical query: {physical_query_id}")
    if row["status"] != "active":
        raise RuntimeError(f"Physical query is not active: {physical_query_id}")
    run = connection.execute(
        """
        SELECT * FROM query_runs
        WHERE provider = 'OpenAlex'
          AND physical_query_id = ?
          AND run_role = ?
        """,
        (physical_query_id, run_role),
    ).fetchone()
    if run is not None and bool(run["complete"]):
        return dict(run)
    if run is not None and run["query_hash"] != row["query_hash"]:
        raise RuntimeError(
            f"Query hash changed during resumable run: {physical_query_id}"
        )
    cursor = str(run["next_cursor"]) if run is not None else "*"
    retrieved_rows = int(run["retrieved_rows"]) if run is not None else 0
    pages = int(run["pages"]) if run is not None else 0
    reported_total = run["reported_total"] if run is not None else None
    keys = openalex_api_keys()
    page_limit = pages + max_pages if max_pages is not None else None
    while cursor:
        api_key = (
            keys[(api_key_offset + pages) % len(keys)] if keys else ""
        )
        url = openalex_url(
            expression=str(row["expression"]),
            filter_expression=str(row["filter_expression"]),
            cursor=cursor,
            per_page=per_page,
            api_key=api_key,
            select=_retrieval_select(),
        )
        try:
            payload = fetcher(url)
        except sqlite3.Error:
            raise
        except Exception as error:
            connection.execute(
                """
                INSERT INTO query_runs(
                    provider, physical_query_id, run_role, query_hash,
                    reported_total, retrieved_rows, unique_hits, pages,
                    next_cursor, complete, stopped_reason, error, updated_at
                ) VALUES (
                    'OpenAlex', ?, ?, ?, ?, ?, 0, ?, ?, 0,
                    'provider_error', ?, ?
                )
                ON CONFLICT(provider, physical_query_id, run_role)
                DO UPDATE SET
                    reported_total = excluded.reported_total,
                    retrieved_rows = excluded.retrieved_rows,
                    pages = excluded.pages,
                    next_cursor = excluded.next_cursor,
                    complete = 0,
                    stopped_reason = 'provider_error',
                    error = excluded.error,
                    updated_at = excluded.updated_at
                """,
                (
                    physical_query_id,
                    run_role,
                    row["query_hash"],
                    reported_total,
                    retrieved_rows,
                    pages,
                    cursor,
                    safe_provider_error(error),
                    utc_now(),
                ),
            )
            connection.commit()
            raise
        results = payload.get("results")
        meta = payload.get("meta")
        if not isinstance(results, list) or not isinstance(meta, dict):
            raise ValueError("Malformed OpenAlex cursor response")
        if reported_total is None and isinstance(meta.get("count"), int):
            reported_total = int(meta["count"])
        page_start = retrieved_rows
        for offset, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                continue
            record = openalex_record(
                item,
                retrieval_route=f"{run_role}:{physical_query_id}",
            )
            insert_openalex_record(connection, record)
            connection.execute(
                """
                INSERT OR IGNORE INTO query_hits(
                    provider, physical_query_id, run_role, record_key, rank
                ) VALUES ('OpenAlex', ?, ?, ?, ?)
                """,
                (
                    physical_query_id,
                    run_role,
                    record["record_key"],
                    page_start + offset,
                ),
            )
        retrieved_rows += len(results)
        pages += 1
        raw_next_cursor = meta.get("next_cursor")
        next_cursor = (
            str(raw_next_cursor)
            if isinstance(raw_next_cursor, str) and raw_next_cursor
            else ""
        )
        unique_hits = connection.execute(
            """
            SELECT COUNT(*) FROM query_hits
            WHERE provider = 'OpenAlex'
              AND physical_query_id = ?
              AND run_role = ?
            """,
            (physical_query_id, run_role),
        ).fetchone()[0]
        provider_shortfall = (
            not next_cursor
            and reported_total is not None
            and retrieved_rows < int(reported_total)
        )
        complete = int(not next_cursor and not provider_shortfall)
        stopped_reason = (
            "provider_early_termination" if provider_shortfall else ""
        )
        connection.execute(
            """
            INSERT INTO query_runs(
                provider, physical_query_id, run_role, query_hash,
                reported_total, retrieved_rows, unique_hits, pages,
                next_cursor, complete, stopped_reason, error, updated_at
            ) VALUES (
                'OpenAlex', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?
            )
            ON CONFLICT(provider, physical_query_id, run_role) DO UPDATE SET
                reported_total = excluded.reported_total,
                retrieved_rows = excluded.retrieved_rows,
                unique_hits = excluded.unique_hits,
                pages = excluded.pages,
                next_cursor = excluded.next_cursor,
                complete = excluded.complete,
                stopped_reason = excluded.stopped_reason,
                error = '',
                updated_at = excluded.updated_at
            """,
            (
                physical_query_id,
                run_role,
                row["query_hash"],
                reported_total,
                retrieved_rows,
                unique_hits,
                pages,
                next_cursor,
                complete,
                stopped_reason,
                utc_now(),
            ),
        )
        connection.commit()
        cursor = next_cursor
        if page_limit is not None and pages >= page_limit and cursor:
            break
    final = connection.execute(
        """
        SELECT * FROM query_runs
        WHERE provider = 'OpenAlex'
          AND physical_query_id = ?
          AND run_role = ?
        """,
        (physical_query_id, run_role),
    ).fetchone()
    if final is None:
        raise RuntimeError("Query run did not create a checkpoint")
    return dict(final)


def retrieve_active_queries(
    connection: sqlite3.Connection,
    run_role: str,
    max_pages_per_query: int | None = None,
    fetcher: Any = fetch_json,
) -> List[Dict[str, Any]]:
    """Retrieve all active physical queries deterministically."""
    query_ids = [
        str(row["physical_query_id"])
        for row in connection.execute(
            """
            SELECT physical_query_id
            FROM physical_queries
            WHERE provider = 'OpenAlex' AND status = 'active'
            ORDER BY physical_query_id
            """
        )
    ]
    return [
        retrieve_physical_query(
            connection,
            query_id,
            run_role,
            max_pages=max_pages_per_query,
            fetcher=fetcher,
        )
        for query_id in query_ids
    ]


def direct_openalex_seed_status(
    doi: str,
    expression: str = "",
    api_key: str = "",
    fetcher: Any = fetch_json,
) -> bool:
    """Check DOI indexability, optionally combined with a search expression."""
    normalized_doi = normalize_doi(doi)
    if not normalized_doi:
        return False
    url = openalex_url(
        expression=expression,
        filter_expression=f"doi:https://doi.org/{normalized_doi}",
        per_page=1,
        api_key=api_key,
        select="id,doi",
    )
    payload = fetcher(url)
    meta = payload.get("meta")
    return isinstance(meta, dict) and int(meta.get("count") or 0) > 0


def batch_openalex_seed_matches(
    dois: Iterable[str],
    expression: str = "",
    api_key: str = "",
    fetcher: Any = fetch_json,
) -> set[str]:
    """Return DOI identities matching one expression in a bounded OR batch."""
    normalized = sorted(
        {
            normalize_doi(doi)
            for doi in dois
            if normalize_doi(doi)
        }
    )
    if not normalized:
        return set()
    if len(normalized) > 40:
        raise ValueError("OpenAlex DOI validation batches are limited to 40")
    url = openalex_url(
        expression=expression,
        filter_expression="doi:" + "|".join(normalized),
        per_page=100,
        api_key=api_key,
        select="doi",
    )
    payload = fetcher(url)
    results = payload.get("results")
    meta = payload.get("meta")
    if not isinstance(results, list) or not isinstance(meta, dict):
        raise ValueError("Malformed OpenAlex DOI batch response")
    return {
        normalize_doi(item.get("doi"))
        for item in results
        if isinstance(item, dict) and normalize_doi(item.get("doi"))
    }


def retrieve_openalex_ids(
    connection: sqlite3.Connection,
    provider_ids: Iterable[str],
    retrieval_route: str,
    fetcher: Any = fetch_json,
) -> int:
    """Fetch specific OpenAlex IDs for backward citation tracking."""
    keys = openalex_api_keys()
    inserted = 0
    for index, provider_id in enumerate(sorted(set(provider_ids))):
        compact = provider_id.rsplit("/", maxsplit=1)[-1]
        parameters: Dict[str, Any] = {}
        if keys:
            parameters["api_key"] = keys[index % len(keys)]
        url = f"{OPENALEX_BASE_URL}/works/{urllib.parse.quote(compact)}"
        if parameters:
            url += "?" + urllib.parse.urlencode(parameters)
        payload = fetcher(url)
        record = openalex_record(payload, retrieval_route)
        before = connection.total_changes
        insert_openalex_record(connection, record)
        inserted += int(connection.total_changes > before)
    connection.commit()
    return inserted


def fetch_openalex_work(
    provider_id: str,
    retrieval_route: str,
    api_key: str = "",
    fetcher: Any = fetch_json,
) -> Dict[str, Any]:
    """Fetch and normalize one OpenAlex work ID."""
    compact = provider_id.rsplit("/", maxsplit=1)[-1]
    parameters: Dict[str, Any] = {}
    if api_key:
        parameters["api_key"] = api_key
    url = f"{OPENALEX_BASE_URL}/works/{urllib.parse.quote(compact)}"
    if parameters:
        url += "?" + urllib.parse.urlencode(parameters)
    return openalex_record(fetcher(url), retrieval_route)


def retrieve_forward_citations(
    connection: sqlite3.Connection,
    source_record_key: str,
    provider_id: str,
    iteration: int,
    cutoff_date: str,
    fetcher: Any = fetch_json,
) -> int:
    """Cursor-page works citing one OpenAlex source work."""
    compact = provider_id.rsplit("/", maxsplit=1)[-1]
    expression = ""
    filter_expression = (
        f"cites:{compact},to_publication_date:{cutoff_date},"
        "type:article|review"
    )
    cursor = "*"
    keys = openalex_api_keys()
    page = 0
    inserted = 0
    while cursor:
        api_key = keys[page % len(keys)] if keys else ""
        url = openalex_url(
            expression=expression,
            filter_expression=filter_expression,
            cursor=cursor,
            api_key=api_key,
            select=_retrieval_select(),
        )
        payload = fetcher(url)
        results = payload.get("results")
        meta = payload.get("meta")
        if not isinstance(results, list) or not isinstance(meta, dict):
            raise ValueError("Malformed forward-citation response")
        for item in results:
            if not isinstance(item, dict):
                continue
            record = openalex_record(
                item,
                f"forward_citation_round_{iteration}",
            )
            insert_openalex_record(connection, record)
            connection.execute(
                """
                INSERT OR IGNORE INTO citation_edges(
                    source_record_key, target_provider_id, direction,
                    iteration, eligibility_status
                ) VALUES (?, ?, 'forward', ?, 'pending_screening')
                """,
                (
                    source_record_key,
                    record["provider_id"],
                    iteration,
                ),
            )
            inserted += 1
        next_cursor = meta.get("next_cursor")
        cursor = str(next_cursor) if next_cursor else ""
        page += 1
        connection.commit()
    return inserted


def title_similarity(left: str, right: str) -> float:
    """Return normalized bibliographic-title similarity."""
    return SequenceMatcher(
        None,
        normalize_text(left),
        normalize_text(right),
    ).ratio()


def _crossref_year(message: Mapping[str, Any]) -> int | None:
    for field in ("published", "published-print", "published-online"):
        value = message.get(field)
        if not isinstance(value, dict):
            continue
        parts = value.get("date-parts")
        if (
            isinstance(parts, list)
            and parts
            and isinstance(parts[0], list)
            and parts[0]
            and isinstance(parts[0][0], int)
        ):
            return int(parts[0][0])
    return None


def validate_crossref_record(
    doi: str,
    fetcher: Any = fetch_json,
) -> Dict[str, Any]:
    """Retrieve Crossref metadata for one DOI."""
    normalized_doi = normalize_doi(doi)
    encoded = urllib.parse.quote(normalized_doi, safe="")
    mailto = local_environment_value("CROSSREF_MAILTO").strip()
    parameters = (
        "?" + urllib.parse.urlencode({"mailto": mailto})
        if mailto and "@" in mailto
        else ""
    )
    payload = fetcher(
        f"{CROSSREF_BASE_URL}/works/{encoded}{parameters}"
    )
    message = payload.get("message")
    if not isinstance(message, dict):
        raise ValueError("Crossref response has no message object")
    titles = message.get("title")
    title = str(titles[0]) if isinstance(titles, list) and titles else ""
    return {
        "doi": normalize_doi(message.get("DOI") or normalized_doi),
        "title": title,
        "year": _crossref_year(message),
        "type": str(message.get("type") or ""),
    }


def _assess_crossref_record(
    row: Mapping[str, Any],
    fetcher: Any,
) -> Dict[str, Any]:
    """Fetch and classify one Crossref record without touching SQLite."""
    doi = normalize_doi(row["doi"])
    try:
        crossref = validate_crossref_record(doi, fetcher)
        similarity = title_similarity(row["title"], crossref["title"])
        year_match = int(
            row["publication_year"] is None
            or crossref["year"] is None
            or int(row["publication_year"]) == int(crossref["year"])
        )
        compatible_types = {
            "article": {"journal-article", "proceedings-article"},
            "review": {"journal-article"},
        }
        type_match = int(
            crossref["type"]
            in compatible_types.get(str(row["work_type"]), set())
        )
        identity_conflicts: List[str] = []
        if normalize_doi(crossref["doi"]) != doi:
            identity_conflicts.append("doi")
        if similarity < 0.85:
            identity_conflicts.append("title")
        if not type_match:
            identity_conflicts.append("type")
        conflicts = [*identity_conflicts]
        if not year_match:
            conflicts.append("year")
        status = (
            "conflict"
            if identity_conflicts
            else "validated_date_variant"
            if not year_match
            else "validated"
        )
        return {
            "status": status,
            "title_match": similarity,
            "year_match": year_match,
            "type_match": type_match,
            "crossref_title": crossref["title"],
            "crossref_year": crossref["year"],
            "crossref_type": crossref["type"],
            "conflict_reason": (
                "year_date_variant"
                if status == "validated_date_variant"
                else "|".join(conflicts)
            ),
        }
    except Exception as error:
        error_summary = safe_provider_error(error)
        not_found = "HTTPError status=404" in error_summary
        return {
            "status": "conflict" if not_found else "error",
            "title_match": None,
            "year_match": None,
            "type_match": None,
            "crossref_title": "",
            "crossref_year": None,
            "crossref_type": "",
            "conflict_reason": (
                "crossref_doi_not_found" if not_found else error_summary
            ),
        }


def _store_crossref_assessment(
    connection: sqlite3.Connection,
    row: Mapping[str, Any],
    assessment: Mapping[str, Any],
) -> None:
    """Persist one already-computed assessment on the main thread."""
    connection.execute(
        """
        INSERT INTO crossref_validation(
            record_key, doi, status, title_match, year_match,
            type_match, crossref_title, crossref_year,
            crossref_type, conflict_reason, validated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(record_key) DO UPDATE SET
            doi = excluded.doi,
            status = excluded.status,
            title_match = excluded.title_match,
            year_match = excluded.year_match,
            type_match = excluded.type_match,
            crossref_title = excluded.crossref_title,
            crossref_year = excluded.crossref_year,
            crossref_type = excluded.crossref_type,
            conflict_reason = excluded.conflict_reason,
            validated_at = excluded.validated_at
        """,
        (
            row["record_key"],
            normalize_doi(row["doi"]),
            assessment["status"],
            assessment["title_match"],
            assessment["year_match"],
            assessment["type_match"],
            assessment["crossref_title"],
            assessment["crossref_year"],
            assessment["crossref_type"],
            assessment["conflict_reason"],
            utc_now(),
        ),
    )


def crossref_validate_scope(
    connection: sqlite3.Connection,
    record_keys: Iterable[str],
    fetcher: Any = fetch_json,
) -> Dict[str, int]:
    """Validate DOI-bearing records and queue metadata conflicts."""
    counts = {
        "validated": 0,
        "validated_date_variant": 0,
        "conflict": 0,
        "missing_doi": 0,
        "error": 0,
    }
    records: List[Dict[str, Any]] = []
    for record_key in record_keys:
        row = connection.execute(
            """
            SELECT record_key, doi, title, publication_year, work_type
            FROM records
            WHERE record_key = ?
            """,
            (record_key,),
        ).fetchone()
        if row is None:
            continue
        doi = normalize_doi(row["doi"])
        if not doi:
            counts["missing_doi"] += 1
            continue
        records.append(dict(row))
    worker_count = min(4, len(records)) if fetcher is fetch_json else 1
    if worker_count > 1:
        executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="crossref",
        )
        assessments = executor.map(
            lambda item: _assess_crossref_record(item, fetcher),
            records,
        )
    else:
        executor = None
        assessments = (
            _assess_crossref_record(item, fetcher) for item in records
        )
    for index, (row, assessment) in enumerate(
        zip(records, assessments),
        start=1,
    ):
        _store_crossref_assessment(connection, row, assessment)
        counts[str(assessment["status"])] += 1
        connection.commit()
        if index % 25 == 0:
            print(
                f"[Crossref] {index}/{len(records)} "
                f"validated={counts['validated']} "
                f"date_variants={counts['validated_date_variant']} "
                f"conflicts={counts['conflict']} errors={counts['error']}",
                flush=True,
            )
    if executor is not None:
        executor.shutdown(wait=True)
    log_event(
        connection,
        "crossref_validation",
        "scope",
        "records",
        counts,
    )
    invalidate_stages(
        connection,
        ("audit_complete",),
        "Crossref validation state changed",
    )
    connection.commit()
    return counts


def reclassify_crossref_not_found(
    connection: sqlite3.Connection,
    record_keys: Iterable[str] | None = None,
) -> int:
    """Route stored deterministic DOI 404s to human metadata review."""
    scope = set(record_keys) if record_keys is not None else None
    candidates = [
        str(row["record_key"])
        for row in connection.execute(
            """
            SELECT record_key FROM crossref_validation
            WHERE status = 'error'
              AND conflict_reason LIKE '%HTTPError status=404%'
            ORDER BY record_key
            """
        )
        if scope is None or str(row["record_key"]) in scope
    ]
    for record_key in candidates:
        connection.execute(
            """
            UPDATE crossref_validation
            SET status = 'conflict',
                conflict_reason = 'crossref_doi_not_found',
                validated_at = ?
            WHERE record_key = ?
            """,
            (utc_now(), record_key),
        )
    if candidates:
        log_event(
            connection,
            "crossref_not_found_reclassification",
            "scope",
            "records",
            {
                "reclassified": len(candidates),
                "rule": (
                    "Stable Crossref DOI endpoint 404; retain the paper and "
                    "route bibliographic identity to H2"
                ),
            },
        )
        invalidate_stages(
            connection,
            ("audit_complete",),
            "Crossref DOI 404s routed to human metadata review",
        )
        connection.commit()
    return len(candidates)


def reclassify_crossref_date_variants(
    connection: sqlite3.Connection,
    record_keys: Iterable[str] | None = None,
) -> int:
    """Reclassify pure year differences without another provider request."""
    scope = set(record_keys) if record_keys is not None else None
    candidates = [
        str(row["record_key"])
        for row in connection.execute(
            """
            SELECT record_key FROM crossref_validation
            WHERE status = 'conflict'
              AND conflict_reason = 'year'
              AND title_match >= 0.85
              AND type_match = 1
            ORDER BY record_key
            """
        )
        if scope is None or str(row["record_key"]) in scope
    ]
    for record_key in candidates:
        connection.execute(
            """
            UPDATE crossref_validation
            SET status = 'validated_date_variant',
                conflict_reason = 'year_date_variant',
                validated_at = ?
            WHERE record_key = ?
            """,
            (utc_now(), record_key),
        )
    if candidates:
        log_event(
            connection,
            "crossref_date_variant_reclassification",
            "scope",
            "records",
            {
                "reclassified": len(candidates),
                "rule": (
                    "DOI endpoint, title similarity >=0.85, and compatible "
                    "type establish identity; year difference is retained "
                    "as an online/issue date variant"
                ),
            },
        )
        invalidate_stages(
            connection,
            ("audit_complete",),
            "Crossref pure-year conflicts reclassified",
        )
        connection.commit()
    return len(candidates)


def export_crossref_conflicts(
    connection: sqlite3.Connection,
    output_path: Path,
    record_keys: Iterable[str] | None = None,
) -> int:
    """Export metadata conflicts for human resolution."""
    scope = set(record_keys) if record_keys is not None else None
    rows = [
        {
            **dict(row),
            "reviewer_role": "",
            "resolution": "",
            "resolution_notes": "",
        }
        for row in connection.execute(
            """
            SELECT v.*, r.title AS openalex_title,
                   r.publication_year AS openalex_year,
                   r.work_type AS openalex_type
            FROM crossref_validation v
            JOIN records r USING(record_key)
            WHERE v.status IN ('conflict', 'error')
            ORDER BY v.record_key
            """
        )
        if scope is None or str(row["record_key"]) in scope
    ]
    fields = (
        list(rows[0])
        if rows
        else [
            "record_key",
            "doi",
            "status",
            "conflict_reason",
            "reviewer_role",
            "resolution",
            "resolution_notes",
        ]
    )
    write_csv(output_path, rows, fields)
    return len(rows)


def import_crossref_resolutions(
    connection: sqlite3.Connection,
    input_path: Path,
) -> int:
    """Record H2-reviewed resolution of OpenAlex/Crossref conflicts."""
    snapshot_path = snapshot_import_file(
        connection,
        input_path,
        "crossref_resolution",
    )
    rows = read_csv(snapshot_path)
    imported = 0
    allowed = {
        "accept_openalex",
        "accept_crossref",
        "manual_bibliographic_resolution",
        "exclude_mapping_error",
    }
    for row in rows:
        record_key = str(row.get("record_key") or "").strip()
        resolution = str(row.get("resolution") or "").strip().casefold()
        if not record_key or not resolution:
            continue
        if str(row.get("reviewer_role") or "").strip().upper() != "H2":
            raise ValueError("Crossref conflicts require H2 resolution")
        if resolution not in allowed:
            raise ValueError(f"Invalid Crossref resolution: {resolution}")
        notes = str(row.get("resolution_notes") or "").strip()
        if not notes:
            raise ValueError("Crossref resolutions require notes")
        exists = connection.execute(
            """
            SELECT 1 FROM crossref_validation
            WHERE record_key = ? AND status IN ('conflict', 'error')
            """,
            (record_key,),
        ).fetchone()
        if exists is None:
            raise ValueError(
                f"Record is not an unresolved Crossref conflict: {record_key}"
            )
        connection.execute(
            """
            UPDATE crossref_validation
            SET status = 'resolved',
                conflict_reason = conflict_reason || ?,
                validated_at = ?
            WHERE record_key = ?
            """,
            (
                f"|H2:{resolution}:{notes}",
                utc_now(),
                record_key,
            ),
        )
        imported += 1
    if imported:
        invalidate_stages(
            connection,
            ("audit_complete",),
            "Crossref conflict resolutions changed",
        )
    connection.commit()
    return imported


def query_definition_hash(
    expression: str,
    filter_expression: str,
) -> str:
    """Hash only non-secret provider request semantics."""
    return json_hash(
        {
            "provider": "OpenAlex",
            "expression": expression,
            "filter": filter_expression,
        }
    )
